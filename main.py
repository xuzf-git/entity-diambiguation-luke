import os
import sys
import yaml
import json
import time
import logging
import argparse
import functools
import subprocess

logging.getLogger("transformers").setLevel(logging.WARNING)

import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from transformers import WEIGHTS_NAME
from luke.utils.model_utils import ModelArchive
from luke.utils.entity_vocab import MASK_TOKEN, PAD_TOKEN
from entity_disambiguation.model import LukeForEntityDisambiguation
from utils import set_seed
from utils.evaluate import evaluate
from utils.trainer import Trainer, trainer_args
from utils.dataset import EntityDisambiguationDataset, convert_documents_to_features

import wandb

logger = logging.getLogger(__name__)
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"


def main():
    wandb.init(project="reproduce-luke-ed", entity="xuzf")
    
    parser = argparse.ArgumentParser()
    # parameters
    ## experiment path
    parser.add_argument("--model-file", type=str, help='path of pre-tained model weights and config')
    parser.add_argument("--data-dir", type=str, help='path of training or evaluate data')
    parser.add_argument("--output-dir", default="./output/results/exp_" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), help='output directory')
    ## train/eval/test scope
    parser.add_argument("--do_train", action='store_true', help='finetune model on Conll2003')
    parser.add_argument("--do_eval", action='store_true', help='eval on testb')
    parser.add_argument("--do_test", action='store_true', help='test on the chosen datasets')
    parser.add_argument("--test-set", nargs='+', help='multi-select the test dataset', choices=["test_b", "test_b_ppr", "ace2004", "aquaint", "msnbc", "wikipedia", "clueweb"])
    ## hardware setting
    parser.add_argument("--master-port", type=int)
    parser.add_argument("--num-gpus", type=int, help='the number of gpus')
    parser.add_argument("--local-rank",
                        type=int,
                        help='local_rank is used in the case of multiple gpus, so local_rank is valid only when num_gpus > 0.\
                        it exactly represents the GPU index used in the current process,\
                        (local_rank=-1) means to use the default cuda; (local_rank>=0) means to use corresponding GPU.')
    ## hyperparameter
    parser.add_argument("--num-train-epochs", type=int)
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--max-candidate-length", type=int)
    parser.add_argument("--masked-entity-prob", type=int)
    parser.add_argument("--document-split-mode", choices=["simple", "per_mention"])
    parser.add_argument("--update-entity-emb", action='store_true', help='default fixed entity embedding')
    parser.add_argument("--update-entity-bias", action='store_true', help='default fixed entity bias')
    parser.add_argument("--seed")
    parser.add_argument("--no-context-entities", action='store_true', help='context entities is used by default')
    parser.add_argument("--context-entity-selection-order", choices=["natural", "random", "highest_prob"], help='order of entity disambiguation')

    # 默认为 config.json 中的配置，命令行参数优先级更高
    with open('./config.yaml', 'r') as fconfig:
        args_config = yaml.load(fconfig, Loader=yaml.FullLoader)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if not value is None and not value is False:
            args_config[key] = value
    with open('./config.yaml', 'w') as fconfig:
        yaml.dump(args_config, fconfig)
    args = argparse.Namespace(**args_config)

    # 记录当前 run 的参数设置
    wandb.config.update(args)

    if args.local_rank == -1 and args.num_gpus > 1:  ## 单机多卡——进程处理
        current_env = os.environ.copy()
        current_env["MASTER_ADDR"] = "127.0.0.1"
        current_env["MASTER_PORT"] = str(args.master_port)
        current_env["WORLD_SIZE"] = str(args.num_gpus)

        processes = []

        # local_rank 从 0 到 num_gpus 创建 num_gpus 个子进程
        for args.local_rank in range(0, args.num_gpus):
            current_env["RANK"] = str(args.local_rank)
            current_env["LOCAL_RANK"] = str(args.local_rank)

            cmd = [sys.executable, "-u", "-m", "main.py", "--local-rank={}".format(args.local_rank)]
            cmd.extend(sys.argv[1:])

            process = subprocess.Popen(cmd, env=current_env)
            processes.append(process)

        for process in processes:
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)

        sys.exit(0)
    else:  ## 单卡进程处理
        if args.local_rank not in (-1, 0):
            logging.basicConfig(format=LOG_FORMAT, level=logging.WARNING)
        else:
            logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)

        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
            logger.info("Output dir: %s", args.output_dir)

        if args.num_gpus == 0:
            args.device = torch.device("cpu")
        elif args.local_rank == -1:
            args.device = torch.device("cuda")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.local_rank)
            args.device = torch.device("cuda")

        if args.model_file:
            model_archive = ModelArchive.load(args.model_file)
            args.entity_vocab = model_archive.entity_vocab
            args.bert_model_name = model_archive.bert_model_name
            if model_archive.bert_model_name.startswith("roberta"):
                # the current example code does not support the fast tokenizer
                args.tokenizer = RobertaTokenizer.from_pretrained(model_archive.bert_model_name)
            else:
                args.tokenizer = model_archive.tokenizer
            args.model_config = model_archive.config
            args.max_mention_length = model_archive.max_mention_length
            args.model_weights = model_archive.state_dict

    run(args=args)


def run(args:argparse.Namespace):

    set_seed(args.seed)

    # 读入数据集
    dataset = EntityDisambiguationDataset(args.data_dir)
    # 获取全部实体的title
    entity_titles = []
    for data in dataset.get_all_datasets():
        for document in data:
            for mention in document.mentions:
                entity_titles.append(mention.title)
                for candidate in mention.candidates:
                    entity_titles.append(candidate.title)
    entity_titles = frozenset(entity_titles)
    # 构建entity词典
    entity_vocab = {PAD_TOKEN: 0, MASK_TOKEN: 1}
    for n, title in enumerate(sorted(entity_titles), 2):
        entity_vocab[title] = n
    # 根据 model_config 和 args 构建模型
    model_config = args.model_config
    model_config.entity_vocab_size = len(entity_vocab)
    model_weights = args.model_weights
    orig_entity_vocab = args.entity_vocab
    orig_entity_emb = model_weights["entity_embeddings.entity_embeddings.weight"]
    if orig_entity_emb.size(0) != len(entity_vocab):  # detect whether the model is fine-tuned
        entity_emb = orig_entity_emb.new_zeros((len(entity_titles) + 2, model_config.hidden_size))
        orig_entity_bias = model_weights["entity_predictions.bias"]
        entity_bias = orig_entity_bias.new_zeros(len(entity_titles) + 2)
        for title, index in entity_vocab.items():
            if title in orig_entity_vocab:
                orig_index = orig_entity_vocab[title]
                entity_emb[index] = orig_entity_emb[orig_index]
                entity_bias[index] = orig_entity_bias[orig_index]
        model_weights["entity_embeddings.entity_embeddings.weight"] = entity_emb
        model_weights["entity_embeddings.mask_embedding"] = entity_emb[1].view(1, -1)
        model_weights["entity_predictions.decoder.weight"] = entity_emb
        model_weights["entity_predictions.bias"] = entity_bias
        del orig_entity_bias, entity_emb, entity_bias
    del orig_entity_emb
    model = LukeForEntityDisambiguation(model_config)
    model.load_state_dict(model_weights, strict=False)
    model.to(args.device)

    wandb.watch(model, log='all', log_graph=True)

    def collate_fn(batch, is_eval=False):
        def create_padded_sequence(attr_name, padding_value):
            tensors = [torch.tensor(getattr(o, attr_name), dtype=torch.long) for o in batch]
            return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)

        ret = dict(
            word_ids=create_padded_sequence("word_ids", args.tokenizer.pad_token_id),
            word_segment_ids=create_padded_sequence("word_segment_ids", 0),
            word_attention_mask=create_padded_sequence("word_attention_mask", 0),
            entity_ids=create_padded_sequence("entity_ids", 0),
            entity_position_ids=create_padded_sequence("entity_position_ids", -1),
            entity_segment_ids=create_padded_sequence("entity_segment_ids", 0),
            entity_attention_mask=create_padded_sequence("entity_attention_mask", 0),
        )
        ret["entity_candidate_ids"] = create_padded_sequence("entity_candidate_ids", 0)

        if is_eval:
            ret["document"] = [o.document for o in batch]
            ret["mentions"] = [o.mentions for o in batch]
            ret["target_mention_indices"] = [o.target_mention_indices for o in batch]

        return ret

    if args.do_train:
        train_data = convert_documents_to_features(
            dataset.train,
            args.tokenizer,
            entity_vocab,
            "train",
            "simple",
            args.max_seq_length,
            args.max_candidate_length,
            args.max_mention_length,
        )
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size, collate_fn=collate_fn, shuffle=True)

        logger.info("Update entity embeddings during training: %s", args.update_entity_emb)
        if not args.update_entity_emb:
            model.entity_embeddings.entity_embeddings.weight.requires_grad = False
        logger.info("Update entity bias during training: %s", args.update_entity_bias)
        if not args.update_entity_bias:
            model.entity_predictions.bias.requires_grad = False

        num_train_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        trainer = EntityDisambiguationTrainer(args, model, train_dataloader, num_train_steps)
        trainer.train()

        if args.output_dir:
            logger.info("Saving model to %s", args.output_dir)
            torch.save(model.state_dict(), os.path.join(args.output_dir, WEIGHTS_NAME))
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    results = {}

    if args.do_eval:
        model.eval()

        for dataset_name in set(args.test_set):
            print("***** Dataset: %s *****" % dataset_name)
            eval_documents = getattr(dataset, dataset_name)
            eval_data = convert_documents_to_features(
                eval_documents,
                args.tokenizer,
                entity_vocab,
                "eval",
                args.document_split_mode,
                args.max_seq_length,
                args.max_candidate_length,
                args.max_mention_length,
            )
            eval_dataloader = DataLoader(eval_data, batch_size=1, collate_fn=functools.partial(collate_fn, is_eval=True))
            predictions_file = None
            if args.output_dir:
                predictions_file = os.path.join(args.output_dir, "eval_predictions_%s.jsonl" % dataset_name)
            results[dataset_name] = evaluate(args, eval_dataloader, model, entity_vocab, predictions_file)

        if args.output_dir:
            output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as f:
                json.dump(results, f, indent=2, sort_keys=True)

    return results



class EntityDisambiguationTrainer(Trainer):
    def _create_model_arguments(self, batch):
        batch["entity_labels"] = batch["entity_ids"].clone()
        for index, entity_length in enumerate(batch["entity_attention_mask"].sum(1).tolist()):
            masked_entity_length = max(1, round(entity_length * self.args.masked_entity_prob))
            permutated_indices = torch.randperm(entity_length)[:masked_entity_length]
            batch["entity_ids"][index, permutated_indices[:masked_entity_length]] = 1  # [MASK]
            batch["entity_labels"][index, permutated_indices[masked_entity_length:]] = -1

        return batch


if __name__ == "__main__":
    main()