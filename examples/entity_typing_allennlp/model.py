from typing import List, Dict
import torch
import torch.nn as nn

from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.training.metrics import CategoricalAccuracy
from examples.utils.embedders.luke_embedder import PretrainedLukeEmbedder
from examples.utils.embedders.transformers_luke_embedder import TransformersLukeEmbedder

from .modules.feature_extractor import ETFeatureExtractor


from .metrics.multiway_f1 import MultiwayF1


@Model.register("entity_typing")
class EntityTypeClassifier(Model):
    """
    Model based on
    ``Matching the Blanks: Distributional Similarity for Relation Learning``
    (https://www.aclweb.org/anthology/P19-1279/)
    """

    def __init__(
        self,
        vocab: Vocabulary,
        feature_extractor: ETFeatureExtractor,
        dropout: float = 0.1,
        label_name_space: str = "labels",
        text_field_key: str = "tokens",
        feature_type: str = "entity_start",
        ignored_labels: List[str] = None,
    ):

        super().__init__(vocab=vocab)
        self.feature_extractor = feature_extractor
        self.classifier = nn.Linear(self.feature_extractor.get_output_dim(), vocab.get_vocab_size(label_name_space))

        self.text_field_key = text_field_key
        self.label_name_space = label_name_space

        self.dropout = nn.Dropout(p=dropout)
        self.criterion = nn.CrossEntropyLoss()

        self.metrics = {
            "accuracy": CategoricalAccuracy(),
        }
        self.f1_score = MultiwayF1(ignored_labels=ignored_labels)

    def forward(
        self,
        word_ids: TextFieldTensors,
        entity_span: torch.LongTensor,
        label: torch.LongTensor = None,
        entity_ids: torch.LongTensor = None,
        input_sentence: List[str] = None,
        **kwargs,
    ):
        feature_vector = self.feature_extractor(word_ids[self.text_field_key], entity_span, entity_ids)
        feature_vector = self.dropout(feature_vector)
        logits = self.classifier(feature_vector)

        prediction_logits, prediction = logits.max(dim=-1)

        output_dict = {
            "input": input_sentence,
            "prediction": prediction,
        }

        if label is not None:
            output_dict["loss"] = self.criterion(logits, label)
            output_dict["gold_label"] = label
            self.metrics["accuracy"](logits, label)

            prediction_labels = [
                self.vocab.get_token_from_index(i, namespace=self.label_name_space) for i in prediction.tolist()
            ]
            gold_labels = [self.vocab.get_token_from_index(i, namespace=self.label_name_space) for i in label.tolist()]
            self.f1_score(prediction, label, prediction_labels, gold_labels)

        return output_dict

    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]):
        output_dict["prediction"] = self.make_label_human_readable(output_dict["prediction"])

        if "gold_label" in output_dict:
            output_dict["gold_label"] = self.make_label_human_readable(output_dict["gold_label"])
        return output_dict

    def make_label_human_readable(self, label: torch.Tensor):
        return [self.vocab.get_token_from_index(i.item(), namespace=self.label_name_space) for i in label]

    def get_metrics(self, reset: bool = False):
        output_dict = {k: metric.get_metric(reset=reset) for k, metric in self.metrics.items()}
        output_dict.update(self.f1_score.get_metric(reset))
        return output_dict
