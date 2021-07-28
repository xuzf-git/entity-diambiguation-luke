import click
import json
import torch
import tqdm
import logging
import sys

sys.path.append("./")

from examples_allennlp.bucc import read_gold_sentence_pairs
from examples.utils.retrieval.scoring_functions import ScoringFunction
from examples.utils.retrieval.retrievers import Retriever
from examples.utils.retrieval.metrics import compute_f1_score
from examples.utils.retrieval.models import Seq2VecEncoder

from allennlp.nn import util as nn_util
from allennlp.common.util import import_module_and_submodules
from allennlp.data import DatasetReader, Vocabulary, DataLoader
from allennlp.common.params import Params, _environment_variables


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_sentence_embeddings(data_loader, model, device: torch.device, debug: bool = False):
    sentence_embeddings = []
    sentences = []
    for batch in tqdm.tqdm(data_loader):
        batch = nn_util.move_to_device(batch, device)
        sentences += batch.pop("sentence")
        if debug:
            sentence_embeddings.append(torch.rand(len(batch), 1))
        else:
            output_embeddings = model(**batch).detach()
            sentence_embeddings.append(output_embeddings)
    sentence_embeddings = torch.cat(sentence_embeddings, dim=0)
    return sentence_embeddings, sentences


def sharding(iterable, sharding_size: int = 8192):
    l = len(iterable)
    for ndx in range(0, l, sharding_size):
        yield iterable[ndx : min(ndx + sharding_size, l)]


@click.command()
@click.argument("config-path")
@click.argument("bucc-source-data-path")
@click.argument("bucc-target-data-path")
@click.argument("bucc-gold-data-path")
@click.option("--output-file", default=None)
@click.option("--scoring-function", default="cosine")
@click.option("--retriever", default="simple")
@click.option("--backward", is_flag=True)
@click.option("--cuda-device", default=-1)
@click.option("--scoring-sharding-size", type=int, default=512)
@click.option("--retrieval-sharding-size", type=int, default=8192)
@click.option("--debug", is_flag=True)
@click.option("--overrides", type=str, default=None)
@torch.no_grad()
def evaluate_bucc(
    config_path: str,
    bucc_source_data_path: str,
    bucc_target_data_path: str,
    bucc_gold_data_path: str,
    output_file: str,
    scoring_function: str,
    retriever: str,
    backward: bool,
    cuda_device: int,
    scoring_sharding_size: int,
    retrieval_sharding_size: int,
    debug: bool,
    overrides: str,
):

    config_params = Params.from_file(config_path, ext_vars=_environment_variables(), params_overrides=overrides)
    import_module_and_submodules("examples")

    if "dataset_reader" in config_params:
        source_reader = DatasetReader.from_params(config_params.pop("dataset_reader"))
        target_reader = source_reader
    elif "source_dataset_reader" in config_params and "target_dataset_reader" in config_params:
        source_reader = DatasetReader.from_params(config_params.pop("source_dataset_reader"))
        target_reader = DatasetReader.from_params(config_params.pop("target_dataset_reader"))

    source_data_loader = DataLoader.from_params(
        reader=source_reader, data_path=bucc_source_data_path, params=config_params["data_loader"].duplicate()
    )
    target_data_loader = DataLoader.from_params(
        reader=target_reader, data_path=bucc_target_data_path, params=config_params["data_loader"].duplicate()
    )

    vocab = Vocabulary.from_instances(source_data_loader.iter_instances(), **config_params["vocabulary"])
    vocab.extend_from_vocab(
        Vocabulary.from_instances(target_data_loader.iter_instances(), **config_params["vocabulary"])
    )

    source_data_loader.index_with(vocab)
    target_data_loader.index_with(vocab)

    if "model" in config_params:
        source_model = Seq2VecEncoder.from_params(vocab=vocab, params=config_params.pop("model"))
        target_model = source_model
    else:
        source_model = Seq2VecEncoder.from_params(vocab=vocab, params=config_params.pop("source_model"))
        target_model = Seq2VecEncoder.from_params(vocab=vocab, params=config_params.pop("target_model"))

    if cuda_device > -1:
        device = torch.device(f"cuda:{cuda_device}")
    else:
        device = torch.device("cpu")
    source_model.eval()
    target_model.eval()
    source_model.to(device)
    target_model.to(device)

    gold_sentence_pairs = read_gold_sentence_pairs(bucc_source_data_path, bucc_target_data_path, bucc_gold_data_path)

    logger.info("Extracting embeddings from source...")
    source_embeddings, source_sentences = extract_sentence_embeddings(
        source_data_loader, source_model, device=device, debug=debug
    )
    logger.info("Extracting embeddings from target...")
    target_embeddings, target_sentences = extract_sentence_embeddings(
        target_data_loader, target_model, device=device, debug=debug
    )

    logger.info("Calculating scores...")
    scoring_function = ScoringFunction.by_name(scoring_function)(sharding_size=scoring_sharding_size)
    retriever = Retriever.by_name(retriever)()

    forward_prediction = []
    forward_scores = []
    for source_embedding_shard, source_sentences_shard in zip(
        sharding(source_embeddings, retrieval_sharding_size), sharding(source_sentences, retrieval_sharding_size)
    ):
        scores = scoring_function(source_embedding_shard, target_embeddings)

        max_scores, retrieved_indices = retriever(scores)
        forward_scores += max_scores.tolist()
        retrieved_target_sentences = [target_sentences[i] for i in retrieved_indices]
        forward_prediction += [(src, tgt) for src, tgt in zip(source_sentences_shard, retrieved_target_sentences)]

    if backward:
        backward_prediction = []
        backward_scores = []
        for target_embedding_shard, target_sentences_shard in zip(
            sharding(target_embeddings, retrieval_sharding_size), sharding(target_sentences, retrieval_sharding_size)
        ):
            scores = scoring_function(target_embedding_shard, source_embeddings)

            max_scores, retrieved_indices = retriever(scores)
            backward_scores += max_scores.tolist()
            retrieved_source_sentences = [source_sentences[i] for i in retrieved_indices]
            backward_prediction += [(src, tgt) for tgt, src in zip(target_sentences_shard, retrieved_source_sentences)]

        # take intersection
        final_prediction = []
        final_scores = []
        backward_prediction = set(backward_prediction)
        for pred, score in zip(forward_prediction, forward_scores):
            if pred in backward_prediction:
                final_prediction.append(pred)
                final_scores.append(score)

    else:
        final_prediction = forward_prediction
        final_scores = forward_scores

    sorted_predictions = reversed(sorted(zip(final_scores, final_prediction)))
    filtered_prediction = [prediction for _, prediction in sorted_predictions][: len(gold_sentence_pairs)]
    metrics = compute_f1_score(prediction=filtered_prediction, gold=gold_sentence_pairs)
    retrieve_all_metrics = compute_f1_score(prediction=final_prediction, gold=gold_sentence_pairs)
    logger.info("metrics")
    logger.info(metrics)
    logger.info("retrieve_all_metrics")
    logger.info(retrieve_all_metrics)

    if output_file is not None:
        prediction_dict_list = [
            {"source": src, "target": tgt, "score": score}
            for (src, tgt), score in zip(final_prediction, final_scores)
        ]
        result_dict = {
            "prediction": prediction_dict_list,
            "metrics": metrics,
            "retrieve_all_metrics": retrieve_all_metrics,
        }

        with open(output_file, "w") as f:
            json.dump(result_dict, f, indent=4)


if __name__ == "__main__":
    evaluate_bucc()
