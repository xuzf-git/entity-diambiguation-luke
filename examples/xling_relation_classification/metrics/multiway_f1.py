from typing import List
import numpy as np
import torch
from allennlp.training.metrics import Metric
from collections import Counter

DEFAULT_IGNORED_LABEL = ["no_relation"]


def normalize_kbp37_label(label: str) -> str:
    label = label.replace("(e1,e2)", "")
    label = label.replace("(e2,e1)", "")
    return label


class MultiwayF1(Metric):
    def __init__(self, ignored_labels: List[str] = None, label_normalize_scheme: str = "kbp37"):
        self._false_negatives = Counter()
        self._true_positives = Counter()
        self._false_positives = Counter()

        self.ignored_labels = set(ignored_labels or DEFAULT_IGNORED_LABEL)

        self.label_normalize_scheme = label_normalize_scheme

    def __call__(
        self,
        prediction: torch.LongTensor,
        gold: torch.LongTensor,
        prediction_labels: List[str],
        gold_labels: List[str],
    ):
        prediction, gold = self.detach_tensors(prediction, gold)
        results = prediction == gold

        if self.label_normalize_scheme == "kbp37":
            prediction_labels = [normalize_kbp37_label(l) for l in prediction_labels]
            gold_labels = [normalize_kbp37_label(l) for l in gold_labels]

        for res, pred_label, gold_label in zip(results, prediction_labels, gold_labels):
            if res:
                self._true_positives[pred_label] += 1
            else:
                self._false_negatives[gold_label] += 1
                self._false_positives[pred_label] += 1

    def reset(self):
        self._false_negatives = Counter()
        self._true_positives = Counter()
        self._false_positives = Counter()

    def get_f1(self, label: str) -> float:
        tp = self._true_positives[label]
        if tp == 0:
            return 0

        fn = self._false_negatives[label]
        fp = self._false_positives[label]
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def get_metric(self, reset: bool) -> float:
        all_labels = (
            set(self._false_positives.keys()) | set(self._false_negatives.keys()) | set(self._true_positives.keys())
        )
        macro_average_f1_score = np.mean(
            [self.get_f1(label) for label in all_labels if label not in self.ignored_labels]
        )
        if reset:
            self.reset()
        return macro_average_f1_score
