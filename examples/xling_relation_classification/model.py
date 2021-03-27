from typing import List
import torch
import torch.nn as nn

from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.training.metrics import CategoricalAccuracy
from examples.utils.luke_embedder import PretrainedLukeEmbedder

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
        embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder = None,
        dropout: float = 0.1,
        label_name_space: str = "labels",
        feature_type: str = "entity_start",
        ignored_labels: List[str] = None,
    ):

        super().__init__(vocab=vocab)
        self.embedder = embedder
        self.encoder = encoder

        assert feature_type in {"cls_token", "mention_pooling", "entity_start", "entity_embeddings"}
        self.feature_type = feature_type
        token_embed_size = self.encoder.get_output_dim() if self.encoder else self.embedder.get_output_dim()

        if feature_type == "cls_token":
            feature_size = token_embed_size
        else:
            feature_size = token_embed_size * 2

        self.classifier = nn.Linear(feature_size, vocab.get_vocab_size(label_name_space))

        self.label_name_space = label_name_space

        self.dropout = nn.Dropout(p=dropout)
        self.criterion = nn.CrossEntropyLoss()

        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "f1": MultiwayF1(ignored_labels=ignored_labels),
        }

    def forward(
        self,
        word_ids: TextFieldTensors,
        entity1_span: torch.LongTensor,
        entity2_span: torch.LongTensor,
        labels: torch.LongTensor = None,
        entity_ids: torch.LongTensor = None,
        entity_position_ids: torch.LongTensor = None,
        **kwargs
    ):
        if entity_ids is not None:
            word_ids["tokens"]["entity_ids"] = entity_ids
            word_ids["tokens"]["entity_position_ids"] = entity_position_ids
            word_ids["tokens"]["entity_attention_mask"] = entity_ids == 1

        token_embeddings = self.embedder(word_ids)
        if self.encoder is not None:
            token_embeddings = self.encoder(token_embeddings)

        if self.feature_type == "cls_token":
            feature_vector = token_embeddings[:, 0]
        elif self.feature_type == "mention_pooling":
            entity_1_features = self._span_pooling(token_embeddings, entity1_span)
            entity_2_features = self._span_pooling(token_embeddings, entity2_span)
            feature_vector = torch.cat([entity_1_features, entity_2_features], dim=1)
        elif self.feature_type == "entity_start":
            entity_1_features = self._extract_entity_start(token_embeddings, entity1_span)
            entity_2_features = self._extract_entity_start(token_embeddings, entity2_span)
            feature_vector = torch.cat([entity_1_features, entity_2_features], dim=1)
        elif self.feature_type == "entity_embeddings":
            assert isinstance(self.embedder, PretrainedLukeEmbedder) and self.embedder.output_entity_embeddings
            # token_embeddings is supposed to be a sequence of two entity embeddings
            batch_size, _, embedding_size = token_embeddings.size()
            feature_vector = token_embeddings.view(batch_size, embedding_size * 2)

        feature_vector = self.dropout(feature_vector)
        logits = self.classifier(feature_vector)
        prediction_logits, prediction = logits.max(dim=-1)

        output_dict = {"logits": logits, "prediction": prediction}

        if labels is not None:
            output_dict["loss"] = self.criterion(logits, labels)
            self.metrics["accuracy"](logits, labels)

            prediction_labels = [
                self.vocab.get_token_from_index(i, namespace=self.label_name_space) for i in prediction.tolist()
            ]
            gold_labels = [self.vocab.get_token_from_index(i, namespace=self.label_name_space) for i in labels.tolist()]
            self.metrics["f1"](prediction, labels, prediction_labels, gold_labels)

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
