import click
import _jsonnet
import json
import torch
import tqdm
import logging


from multilingual.tasks.bucc.reader import parse_bucc_file
from multilingual.retrieval.scoring_functions import ScoringFunction
from multilingual.retrieval.retrievers import Retriever
from multilingual.retrieval.metrics.f1_score import compute_f1_score
from multilingual.retrieval.models.seq2vec_encoder import Seq2VecEncoder

from allennlp.nn import util as nn_util
from allennlp.common.util import import_module_and_submodules
from allennlp.data import DatasetReader, Vocabulary, DataLoader
from allennlp.common.params import Params, _environment_variables

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_sentence_embeddings(data_loader, model, device: torch.device, debug: bool=False):
    sentence_embeddings = []
    indices = []
    for batch in tqdm.tqdm(data_loader):
        batch = nn_util.move_to_device(batch, device)
        indices += batch.pop("index")
        if debug:
            sentence_embeddings.append(torch.rand(len(batch), 1))
        else:
            output_embeddings = model(**batch)
            sentence_embeddings.append(output_embeddings)
    sentence_embeddings = torch.cat(sentence_embeddings, dim=0)
    return sentence_embeddings, indices


@click.command()
@click.argument("config-path")
@click.argument("bucc-source-data-path")
@click.argument("bucc-target-data-path")
@click.argument("bucc-gold-data-path")
@click.option("--output-file", default=None)
@click.option("--scoring-function", default="cosine")
@click.option("--retriever", default="margin")
@click.option("--cuda-device", default=-1)
@click.option("--debug", is_flag=True)
@torch.no_grad()
def evaluate_bucc(
    config_path: str,
    bucc_source_data_path: str,
    bucc_target_data_path: str,
    bucc_gold_data_path: str,
    output_file: str,
    scoring_function: str,
    retriever: str,
    cuda_device: int,
    debug: bool,
):

    config_params = Params(json.loads(_jsonnet.evaluate_file(config_path, ext_vars=_environment_variables())))
    import_module_and_submodules("multilingual")

    if "dataset_reader" in config_params:
        reader = DatasetReader.from_params(config_params.pop("dataset_reader"))
        source_dataset = reader.read(bucc_source_data_path)
        target_dataset = reader.read(bucc_target_data_path)

    elif "source_dataset_reader" in config_params and "target_dataset_reader" in config_params:
        source_reader = DatasetReader.from_params(config_params.pop("source_dataset_reader"))
        target_reader = DatasetReader.from_params(config_params.pop("target_dataset_reader"))
        source_dataset = source_reader.read(bucc_source_data_path)
        target_dataset = target_reader.read(bucc_target_data_path)

    vocab = Vocabulary.from_instances(source_dataset, **config_params["vocabulary"])
    vocab.extend_from_vocab(Vocabulary.from_instances(target_dataset, **config_params["vocabulary"]))
    source_dataset.index_with(vocab)
    target_dataset.index_with(vocab)

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
    source_model.to(device)
    target_model.to(device)

    gold_indices = [idx_pair for idx_pair in parse_bucc_file(bucc_gold_data_path)]

    source_data_loader = DataLoader.from_params(dataset=source_dataset, params=config_params["data_loader"].duplicate())
    target_data_loader = DataLoader.from_params(dataset=target_dataset, params=config_params["data_loader"].duplicate())

    logger.info("Extracting embeddings from source...")
    source_embeddings, source_indices = extract_sentence_embeddings(
        source_data_loader, source_model, device=device, debug=debug
    )
    logger.info("Extracting embeddings from target...")
    target_embeddings, target_indices = extract_sentence_embeddings(
        target_data_loader, target_model, device=device, debug=debug
    )

    logger.info("Calculating scores...")
    scoring_function = ScoringFunction.by_name(scoring_function)()
    scores = scoring_function(source_embeddings, target_embeddings)

    retriever = Retriever.by_name(retriever)()
    max_scores, retrieved_indices = retriever(scores)
    max_scores = max_scores.tolist()
    retrieved_target_indices = [target_indices[i] for i in retrieved_indices]
    prediction = [(src, tgt) for src, tgt in zip(source_indices, retrieved_target_indices)]

    sorted_predictions = reversed(sorted(zip(max_scores, prediction)))
    filtered_prediction = [src_tgt for _, src_tgt in sorted_predictions][: len(gold_indices)]
    metrics = compute_f1_score(prediction=filtered_prediction, gold=gold_indices)
    retrieve_all_metrics = compute_f1_score(prediction=prediction, gold=gold_indices)
    logger.info("metrics")
    logger.info(metrics)
    logger.info("retrieve_all_metrics")
    logger.info(retrieve_all_metrics)

    # find best threshold
    if output_file is not None:
        prediction_dict_list = [
            {"source": src, "target": tgt, "score": score} for (src, tgt), score in zip(prediction, max_scores)
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
