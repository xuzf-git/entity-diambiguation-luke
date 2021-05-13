from typing import List
import torch
import torch.nn as nn

from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.training.metrics import CategoricalAccuracy

from examples.utils.embedders.luke_embedder import PretrainedLukeEmbedder

from .metrics.span_to_label_f1 import SpanToLabelF1


@Model.register("exhausitce_ner")
class ExhaustiveNERModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        embedder: TokenEmbedder,
        encoder: Seq2SeqEncoder = None,
        dropout: float = 0.1,
        label_name_space: str = "labels",
        text_field_key: str = "tokens",
        combine_word_and_entity_features: bool = False,
    ):
        super().__init__(vocab=vocab)
        self.embedder = embedder
        self.encoder = encoder

        self.combine_word_and_entity_features = combine_word_and_entity_features
        if combine_word_and_entity_features and not self.is_using_luke_with_entity():
            raise ValueError(f"You need use PretrainedLukeEmbedder and set output_entity_embeddings True.")

        token_embed_size = self.encoder.get_output_dim() if self.encoder else self.embedder.get_output_dim()
        if self.is_using_luke_with_entity():
            if self.combine_word_and_entity_features:
                feature_size = token_embed_size * 3
            else:
                feature_size = token_embed_size
        else:
            feature_size = token_embed_size * 2

        self.text_field_key = text_field_key
        self.classifier = nn.Linear(feature_size, vocab.get_vocab_size(label_name_space))

        self.dropout = nn.Dropout(p=dropout)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

        self.span_f1 = SpanToLabelF1()
        self.span_accuracy = CategoricalAccuracy()

    def is_using_luke_with_entity(self) -> bool:
        # check if the token embedder is Luke
        return isinstance(self.embedder, PretrainedLukeEmbedder) and self.embedder.output_entity_embeddings

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
        **kwargs,
    ):

        if entity_ids is not None:
            assert self.is_using_luke_with_entity()
            word_ids[self.text_field_key]["entity_ids"] = entity_ids
            word_ids[self.text_field_key]["entity_position_ids"] = entity_position_ids
            word_ids[self.text_field_key]["entity_attention_mask"] = entity_ids == 1

        if self.is_using_luke_with_entity():
            assert entity_ids is not None
            token_embeddings, entity_embeddings = self.embedder(**word_ids[self.text_field_key])
        else:
            token_embeddings = self.embedder(**word_ids[self.text_field_key])
            entity_embeddings = None

        if self.encoder is not None:
            token_embeddings = self.encoder(token_embeddings)
        embedding_size = token_embeddings.size(-1)

        entity_start_positions = entity_start_positions.unsqueeze(-1).expand(-1, -1, embedding_size)
        start_embeddings = torch.gather(token_embeddings, -2, entity_start_positions)

        entity_end_positions = entity_end_positions.unsqueeze(-1).expand(-1, -1, embedding_size)
        end_embeddings = torch.gather(token_embeddings, -2, entity_end_positions)

        word_feature_vector = torch.cat([start_embeddings, end_embeddings], dim=2)

        if self.combine_word_and_entity_features:
            feature_vector = torch.cat([word_feature_vector, entity_embeddings], dim=2)
        elif self.is_using_luke_with_entity():
            feature_vector = entity_embeddings
        else:
            feature_vector = word_feature_vector

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
