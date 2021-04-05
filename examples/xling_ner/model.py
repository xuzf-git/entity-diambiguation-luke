from typing import List
import torch
import torch.nn as nn

from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.training.metrics import CategoricalAccuracy

from examples.utils.embedders.luke_embedder import PretrainedLukeEmbedder

from .metrics.span_to_label_f1 import SpanToLabelF1


@Model.register("exhausitce_ner")
class ExhaustiveNERModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder = None,
        dropout: float = 0.1,
        label_name_space: str = "labels",
    ):
        super().__init__(vocab=vocab)
        self.embedder = embedder
        self.encoder = encoder

        token_embed_size = self.encoder.get_output_dim() if self.encoder else self.embedder.get_output_dim()
        feature_size = token_embed_size if self.is_using_luke_with_entity() else token_embed_size * 2
        self.classifier = nn.Linear(feature_size, vocab.get_vocab_size(label_name_space))

        self.dropout = nn.Dropout(p=dropout)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

        self.span_f1 = SpanToLabelF1()
        self.span_accuracy = CategoricalAccuracy()

    def is_using_luke_with_entity(self) -> bool:
        # check if the token embedder is Luke
        token_embedder = self.embedder._token_embedders["tokens"]
        return isinstance(token_embedder, PretrainedLukeEmbedder) and token_embedder.output_entity_embeddings

    def forward(
        self,
        word_ids: TextFieldTensors,
        entity_start_positions: torch.LongTensor,
        entity_end_positions: torch.LongTensor,
        original_entity_spans: torch.LongTensor,
        doc_id: List[str],
        labels: torch.LongTensor = None,
        entity_ids: torch.LongTensor = None,
        entity_position_ids: torch.LongTensor = None,
        **kwargs
    ):

        if entity_ids is not None:
            assert self.is_using_luke_with_entity()
            word_ids["tokens"]["entity_ids"] = entity_ids
            word_ids["tokens"]["entity_position_ids"] = entity_position_ids
            word_ids["tokens"]["entity_attention_mask"] = entity_ids == 1

        token_embeddings = self.embedder(word_ids)
        if self.encoder is not None:
            token_embeddings = self.encoder(token_embeddings)
        embedding_size = token_embeddings.size(-1)

        if entity_ids is None:
            entity_start_positions = entity_start_positions.unsqueeze(-1).expand(-1, -1, embedding_size)
            start_embeddings = torch.gather(token_embeddings, -2, entity_start_positions)

            entity_end_positions = entity_end_positions.unsqueeze(-1).expand(-1, -1, embedding_size)
            end_embeddings = torch.gather(token_embeddings, -2, entity_end_positions)

            feature_vector = torch.cat([start_embeddings, end_embeddings], dim=2)
        else:
            feature_vector = token_embeddings

        feature_vector = self.dropout(feature_vector)
        logits = self.classifier(feature_vector)
        prediction_logits, prediction = logits.max(dim=-1)

        output_dict = {"logits": logits, "prediction": prediction}

        if labels is not None:
            output_dict["loss"] = self.criterion(logits.flatten(0, 1), labels.flatten())
            self.span_accuracy(logits, labels, mask=(labels != -1))
            self.span_f1(prediction, labels, prediction_logits, original_entity_spans, doc_id, self.vocab)

        return output_dict

    def get_metrics(self, reset: bool = False):
        output_dict = self.span_f1.get_metric(reset)
        output_dict["span_accuracy"] = self.span_accuracy.get_metric(reset)
        return output_dict
