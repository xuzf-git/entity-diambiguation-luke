from typing import Dict
import torch.nn as nn

from allennlp.data import DatasetReader, DataLoader, TextFieldTensors, Vocabulary
from allennlp.common import Params

from luke.pretraining.model import LukePretrainingModel
from luke.pretraining.validation_evaluator import ValidationEvaluator

from examples.bucc.reader import parse_bucc_file
from examples.bucc.evaluate import extract_sentence_embeddings
from examples.utils.retrieval.scoring_functions import ScoringFunction
from examples.utils.retrieval.retrievers import Retriever
from examples.utils.retrieval.metrics import compute_f1_score
from examples.utils.retrieval.models.bag_of_embeddings import masked_mean_pooling


class LukeSentenceEmbedding(nn.Module):
    def __init__(self, luke_model: LukePretrainingModel):
        super().__init__()
        self.luke_model = luke_model

    def forward(self, tokens: TextFieldTensors):
        input_dict = tokens["tokens"]

        embedding_output = self.luke_model.embeddings(input_dict["token_ids"], input_dict["type_ids"])

        attention_mask = self.luke_model._compute_extended_attention_mask(
            input_dict["mask"], entity_attention_mask=None
        )
        encoder_outputs = self.luke_model.encoder(
            embedding_output, attention_mask, [None] * self.luke_model.config.num_hidden_layers
        )

        sequence_output = encoder_outputs[0]

        return masked_mean_pooling(sequence_output, input_dict["mask"])


@ValidationEvaluator.register("bucc")
class LukeEvaluatorBUCC(ValidationEvaluator):
    def __init__(
        self,
        dataset_reader: DatasetReader,
        source_data_path: str,
        target_data_path: str,
        gold_data_path: str,
        data_loader_params: Dict,
        scoring_function: str,
        retriever: str,
    ):

        if not dataset_reader.manual_distributed_sharding:
            raise ValueError("Only validation with a single GPU is currently supported,"
                             "and we have to ensure the data is loaded as in a single process "
                             "even with multi-GPU training."
                             "Please set ``manual_distributed_sharding`` True to prevent loading validation data "
                             "in a distributed way.")

        data_loader_params = Params(data_loader_params)
        self.source_data_loader = DataLoader.from_params(
            reader=dataset_reader, data_path=source_data_path, params=data_loader_params.duplicate()
        )
        self.target_data_loader = DataLoader.from_params(
            reader=dataset_reader, data_path=target_data_path, params=data_loader_params.duplicate()
        )

        vocab = Vocabulary.from_instances(self.source_data_loader.iter_instances())
        vocab.extend_from_instances(self.target_data_loader.iter_instances())

        self.source_data_loader.index_with(vocab)
        self.target_data_loader.index_with(vocab)

        self.gold_indices = [idx_pair for idx_pair in parse_bucc_file(gold_data_path)]

        self.scoring_function = ScoringFunction.by_name(scoring_function)()
        self.retriever = Retriever.by_name(retriever)()

    def __call__(self, model) -> float:
        device = next(model.parameters()).device

        model = LukeSentenceEmbedding(model)

        source_embeddings, source_indices = extract_sentence_embeddings(self.source_data_loader, model, device=device)
        target_embeddings, target_indices = extract_sentence_embeddings(self.target_data_loader, model, device=device)

        scores = self.scoring_function(source_embeddings, target_embeddings)
        max_scores, retrieved_indices = self.retriever(scores)
        max_scores = max_scores.tolist()
        retrieved_target_indices = [target_indices[i] for i in retrieved_indices]
        prediction = [(src, tgt) for src, tgt in zip(source_indices, retrieved_target_indices)]

        sorted_predictions = reversed(sorted(zip(max_scores, prediction)))
        filtered_prediction = [src_tgt for _, src_tgt in sorted_predictions][: len(self.gold_indices)]
        metrics = compute_f1_score(prediction=filtered_prediction, gold=self.gold_indices)

        return metrics["f1"]