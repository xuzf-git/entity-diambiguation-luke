from typing import List, Tuple
from collections import defaultdict
from allennlp.training.metrics import Metric
from allennlp.data import Vocabulary
import torch

from seqeval.metrics import f1_score, recall_score, precision_score
from seqeval.scheme import IOB2


class SpanToLabelF1(Metric):
    def __init__(self):
        self.prediction = defaultdict(list)
        self.gold_labels = defaultdict(list)

    def __call__(
        self,
        prediction: torch.Tensor,
        gold_labels: torch.Tensor,
        prediction_scores: torch.Tensor,
        original_entity_spans: torch.Tensor,
        doc_id: List[str],
        vocab: Vocabulary,
    ):
        prediction, gold_labels, prediction_scores, original_entity_spans = self.detach_tensors(
            prediction, gold_labels, prediction_scores, original_entity_spans
        )

        for pred, gold, scores, spans, id_ in zip(
            prediction, gold_labels, prediction_scores, original_entity_spans, doc_id
        ):
            pred = pred.tolist()
            gold = gold.tolist()
            scores = scores.tolist()
            spans = spans.tolist()
            for p, g, score, span in zip(pred, gold, scores, spans):
                if g == -1:
                    continue
                p = vocab.get_token_from_index(p, namespace="labels")
                g = vocab.get_token_from_index(g, namespace="labels")

                self.prediction[id_].append((score, span, p))
                self.gold_labels[id_].append((0, span, g))

    def reset(self):
        self.prediction = defaultdict(list)
        self.gold_labels = defaultdict(list)

    def get_metric(self, reset: bool):
        if not reset:
            return {}

        all_prediction_sequence = []
        all_gold_sequence = []
        for doc_id in self.gold_labels.keys():
            all_prediction_sequence.append(self.span_to_label_sequence(self.prediction[doc_id]))
            all_gold_sequence.append(self.span_to_label_sequence(self.gold_labels[doc_id]))
        return dict(
            f1=f1_score(all_gold_sequence, all_prediction_sequence, scheme=IOB2),
            precision=precision_score(all_gold_sequence, all_prediction_sequence, scheme=IOB2),
            recall=recall_score(all_gold_sequence, all_prediction_sequence, scheme=IOB2),
        )

    @staticmethod
    def span_to_label_sequence(span_and_labels: List[Tuple[float, Tuple[int, int], str]]) -> List[str]:
        sequence_length = max([end for score, (start, end), label in span_and_labels])
        label_sequence = ["O"] * sequence_length
        for score, (start, end), label in sorted(span_and_labels, key=lambda x: -x[0]):
            if label == "O" or any([l != "O" for l in label_sequence[start:end]]):
                continue
            label_sequence[start:end] = ["I-" + label] * (end - start)
            label_sequence[start] = "B-" + label

        return label_sequence
