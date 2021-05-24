from typing import List, Dict
import torch
import torch.nn as nn

from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.training.metrics import CategoricalAccuracy
from examples.utils.embedders.luke_embedder import PretrainedLukeEmbedder

from .metrics.multiway_f1 import MultiwayF1


@Model.register("transformers_relation_classifier")
class TransformersRelationClassifier(Model):
    """
    Model based on
    ``Matching the Blanks: Distributional Similarity for Relation Learning``
    (https://www.aclweb.org/anthology/P19-1279/)
    """

    def __init__(
        self,
        vocab: Vocabulary,
        embedder: TokenEmbedder,
        encoder: Seq2SeqEncoder = None,
        dropout: float = 0.1,
        label_name_space: str = "labels",
        text_field_key: str = "tokens",
        feature_type: str = "entity_start",
        ignored_labels: List[str] = None,
        combine_word_and_entity_features: bool = False,
    ):

        super().__init__(vocab=vocab)
        self.embedder = embedder
        self.encoder = encoder

        self.combine_word_and_entity_features = combine_word_and_entity_features
        if combine_word_and_entity_features and not self.is_using_luke_with_entity():
            raise ValueError(f"You need use PretrainedLukeEmbedder and set output_entity_embeddings True.")

        assert feature_type in {"cls_token", "mention_pooling", "entity_start", "entity_embeddings"}
        self.feature_type = feature_type
        token_embed_size = self.encoder.get_output_dim() if self.encoder else self.embedder.get_output_dim()

        if feature_type == "cls_token":
            word_feature_size = token_embed_size
        else:
            word_feature_size = token_embed_size * 2

        if self.is_using_luke_with_entity():
            if self.combine_word_and_entity_features:
                feature_size = word_feature_size + token_embed_size * 2
            else:
                feature_size = token_embed_size * 2
        else:
            feature_size = word_feature_size

        self.classifier = nn.Linear(feature_size, vocab.get_vocab_size(label_name_space))

        self.text_field_key = text_field_key
        self.label_name_space = label_name_space

        self.dropout = nn.Dropout(p=dropout)
        self.criterion = nn.CrossEntropyLoss()

        self.metrics = {
            "accuracy": CategoricalAccuracy(),
        }
        self.f1_score = MultiwayF1(ignored_labels=ignored_labels)

    def is_using_luke_with_entity(self) -> bool:
        # check if the token embedder is Luke
        return isinstance(self.embedder, PretrainedLukeEmbedder) and self.embedder.output_entity_embeddings

    @staticmethod
    def get_span_max_length(span: torch.LongTensor) -> int:
        return (span[:, 1] - span[:, 0] + 1).max().item()

    def span_to_position_ids(self, span: torch.LongTensor, max_length: int = None) -> torch.LongTensor:
        batch_size = span.size(0)
        max_length = max_length or self.get_span_max_length(span)
        position_ids = span.new_full((batch_size, max_length), fill_value=-1)

        for i, (start, end) in enumerate(span):
            positions = torch.arange(start, end + 1)
            position_ids[i, : len(positions)] = positions
        return position_ids

    def forward(
        self,
        word_ids: TextFieldTensors,
        entity1_span: torch.LongTensor,
        entity2_span: torch.LongTensor,
        label: torch.LongTensor = None,
        entity_ids: torch.LongTensor = None,
        input_sentence: List[str] = None,
        **kwargs,
    ):
        if entity_ids is not None:
            assert self.is_using_luke_with_entity()
            word_ids[self.text_field_key]["entity_ids"] = entity_ids

            max_position_length = max(self.get_span_max_length(entity1_span), self.get_span_max_length(entity2_span))
            entity_position_ids = torch.stack(
                [
                    self.span_to_position_ids(entity1_span, max_position_length),
                    self.span_to_position_ids(entity2_span, max_position_length),
                ],
                dim=1,
            )
            word_ids[self.text_field_key]["entity_position_ids"] = entity_position_ids
            word_ids[self.text_field_key]["entity_attention_mask"] = torch.ones_like(entity_ids)

        if self.is_using_luke_with_entity():
            token_embeddings, entity_embeddings = self.embedder(**word_ids[self.text_field_key])
        else:
            token_embeddings = self.embedder(**word_ids[self.text_field_key])
            entity_embeddings = None

        if self.encoder is not None:
            token_embeddings = self.encoder(token_embeddings)

        if self.feature_type == "cls_token":
            word_feature_vector = token_embeddings[:, 0]
        elif self.feature_type == "mention_pooling":
            entity_1_features = self._span_pooling(token_embeddings, entity1_span)
            entity_2_features = self._span_pooling(token_embeddings, entity2_span)
            word_feature_vector = torch.cat([entity_1_features, entity_2_features], dim=1)
        elif self.feature_type == "entity_start":
            entity_1_features = self._extract_entity_start(token_embeddings, entity1_span)
            entity_2_features = self._extract_entity_start(token_embeddings, entity2_span)
            word_feature_vector = torch.cat([entity_1_features, entity_2_features], dim=1)
        else:
            raise ValueError(f"Invalid feature_type: {self.feature_type}")

        if self.is_using_luke_with_entity():
            # token_embeddings is supposed to be a sequence of two entity embeddings
            batch_size, _, embedding_size = entity_embeddings.size()
            entity_feature_vector = entity_embeddings.view(batch_size, embedding_size * 2)

        if self.combine_word_and_entity_features:
            feature_vector = torch.cat([word_feature_vector, entity_feature_vector], dim=1)
        elif self.is_using_luke_with_entity():
            feature_vector = entity_feature_vector
        else:
            feature_vector = word_feature_vector

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

    def _span_pooling(self, token_embeddings: torch.Tensor, span: torch.LongTensor) -> torch.Tensor:
        """
        Parameters
        ----------
        token_embeddings: (batch_size, sequence_length, feature_size)
        span: (batch_size, 2)

        Returns
        ---------
        pooled_embeddings: (batch_size, feature_size)
        """
        pooled_embeddings = []
        for token_emb, (start, end) in zip(token_embeddings, span):
            # The span indices are as follows and we only pool among the word positions.
            # start, ...    , end
            # <e>,   w0, w1,  </e>
            pooled_emb, _ = token_emb[start + 1 : end].max(dim=0)
            pooled_embeddings.append(pooled_emb)
        return torch.stack(pooled_embeddings)

    def _extract_entity_start(self, token_embeddings: torch.Tensor, span: torch.LongTensor) -> torch.Tensor:
        entity_start_position = span[:, 0]
        batch_size, _, embedding_size = token_embeddings.size()
        range_tensor = torch.arange(batch_size, device=token_embeddings.device)
        start_embeddings = token_embeddings[range_tensor, entity_start_position]
        return start_embeddings

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
