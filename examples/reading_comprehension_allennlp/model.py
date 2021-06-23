from typing import List
import torch
import torch.nn as nn

from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.training.metrics import CategoricalAccuracy, Average
from examples.reading_comprehension_allennlp.metrics.qa_metric import QAMetric


@Model.register("transformers_qa")
class TransformersQAModel(Model):
    """
    Model based on
    ``A BERT Baseline for the Natural Questions``
    (https://arxiv.org/abs/1901.08634)
    """

    def __init__(
        self,
        vocab: Vocabulary,
        embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder = None,
        dropout: float = 0.1,
        answer_type_name_space: str = "answer_type",
        max_sequence_length: int = 512,
        qa_metric: QAMetric = None,
    ):

        super().__init__(vocab=vocab)
        self.embedder = embedder
        self.encoder = encoder

        feature_size = self.encoder.get_output_dim() if self.encoder else self.embedder.get_output_dim()
        self.max_sequence_length = max_sequence_length
        self.span_scoring_layer = nn.Linear(feature_size, 2)

        self.answer_type_name_space = answer_type_name_space
        if self.answer_type_name_space is not None:
            self.answer_type_classifier = nn.Linear(feature_size, vocab.get_vocab_size(answer_type_name_space))

        self.dropout = nn.Dropout(p=dropout)
        self.criterion = nn.CrossEntropyLoss()

        self.metrics = {
            "span_loss": Average(),
            "answer_type_loss": Average(),
            "answer_type_accuracy": CategoricalAccuracy(),
            "span_accuracy": CategoricalAccuracy(),
        }

        self.qa_metric = qa_metric

    def forward(
        self,
        question_with_context: TextFieldTensors,
        answer_span: torch.LongTensor,
        answer_type: torch.LongTensor = None,
        metadata: List = None,
        entity_ids: torch.LongTensor = None,
        entity_position_ids: torch.LongTensor = None,
        entity_segment_ids: torch.LongTensor = None,
        entity_attention_mask: torch.LongTensor = None,
        **kwargs
    ):

        if entity_ids is not None:
            question_with_context["tokens"]["entity_ids"] = entity_ids
            question_with_context["tokens"]["entity_position_ids"] = entity_position_ids
            question_with_context["tokens"]["entity_segment_ids"] = entity_segment_ids
            question_with_context["tokens"]["entity_attention_mask"] = entity_attention_mask

        token_embeddings = self.embedder(question_with_context)
        if self.encoder is not None:
            token_embeddings = self.encoder(token_embeddings)

        # compute logits for span prediction
        # shape: (batch_size, sequence_length, feature_size) -> (batch_size, sequence_length, 2)
        span_start_endscores = self.span_scoring_layer(token_embeddings)

        # shape: (batch_size, sequence_length)
        span_start_scores = span_start_endscores[:, :, 0]
        span_end_scores = span_start_endscores[:, :, 1]

        span_start_prediction_score, span_start_prediction = torch.max(span_start_scores, dim=1)
        span_end_prediction_score, span_end_prediction = torch.max(span_end_scores, dim=1)

        output_dict = {
            "span_start_scores": span_start_scores,
            "span_end_scores": span_end_scores,
            "span_start_prediction_score": span_start_prediction_score - span_start_scores[:, 0],
            "span_start_prediction": span_start_prediction,
            "span_end_prediction_score": span_end_prediction_score - span_end_scores[:, 0],
            "span_end_prediction": span_end_prediction,
        }

        if self.answer_type_name_space is not None:
            # compute logits for answer type prediction
            cls_embeddings = token_embeddings[:, 0]
            answer_type_logits = self.answer_type_classifier(cls_embeddings)
            answer_type_prediction = answer_type_logits.argmax(dim=1)
            output_dict.update(
                {"answer_type_logits": answer_type_logits, "answer_type_prediction": answer_type_prediction}
            )

        if answer_span is not None:
            # predict answer span
            # shape: (batch_size, 2) -> (batch_size * 2)
            flatten_answer_span = torch.cat([answer_span[:, 0], answer_span[:, 1]], dim=0)
            # shape: (batch_size * 2, sequence_length)
            flattened_span_start_endscores = torch.cat([span_start_scores, span_end_scores], dim=0)

            span_loss = self.criterion(flattened_span_start_endscores, flatten_answer_span)
            self.metrics["span_loss"](span_loss.item())
            self.metrics["span_accuracy"](flattened_span_start_endscores, flatten_answer_span)
            output_dict["loss"] = span_loss

            if self.answer_type_name_space is not None and answer_type is not None:
                # predict answer type
                answer_type_loss = self.criterion(answer_type_logits, answer_type)
                self.metrics["answer_type_loss"](span_loss.item())
                self.metrics["answer_type_accuracy"](answer_type_logits, answer_type)
                output_dict["loss"] += answer_type_loss

        if not self.training and self.qa_metric:
            self.qa_metric(output_dict, metadata)

        return output_dict

    def get_metrics(self, reset: bool = False):
        metric_results = {k: metric.get_metric(reset=reset) for k, metric in self.metrics.items()}
        if self.qa_metric is not None:
            metric_results[self.qa_metric.validation_metric_name] = self.qa_metric.get_metric(reset=reset)
        return metric_results