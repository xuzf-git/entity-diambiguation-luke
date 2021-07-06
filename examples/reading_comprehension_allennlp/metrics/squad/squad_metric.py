from typing import NamedTuple, Dict, List
from collections import defaultdict
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from allennlp.training.metrics import Metric
from examples.reading_comprehension_allennlp.metrics.qa_metric import QAMetric
from examples.reading_comprehension_allennlp.metrics.squad.squad_evaluate_v1 import evaluate_from_files


class SQuADPrediction(NamedTuple):
    answer: str
    score: float


@QAMetric.register("squad-v1.1")
class SQuADMetric(Metric):
    def __init__(
        self,
        gold_data_path: str,
        prediction_dump_path: str,
        transformers_tokenizer_name: str,
        validation_metric: str = "f1",
    ):
        super().__init__()
        self.document_predictions = defaultdict(list)
        self.passage_answer_candidates = {}
        self.example_to_language = {}
        assert Path(gold_data_path).exists()
        self.gold_data_path = gold_data_path
        self.prediction_dump_path = prediction_dump_path

        self._tokenizer = AutoTokenizer.from_pretrained(transformers_tokenizer_name)

        self.validation_metric = validation_metric

        self.count = 0

    @property
    def validation_metric_name(self) -> str:
        return self.validation_metric

    def __call__(self, output_dict: Dict[str, torch.Tensor], metadata_list: List[Dict]):
        prediction_score = output_dict["prediction_score"].tolist()
        for metadata, (start, end), score in zip(metadata_list, output_dict["span_prediction"], prediction_score,):

            if start == end == 0:
                answer = None
            else:
                answer = self._tokenizer.convert_tokens_to_string(metadata["input_tokens"][start : end + 1])
            self.document_predictions[metadata["example_id"]].append(SQuADPrediction(answer, score))

    def get_metric(self, reset: bool) -> float:
        if not reset:
            return 0

        prediction_dict = {}
        for example_id, predictions in self.document_predictions.items():
            prediction, score = max(predictions, key=lambda x: x.score)
            if prediction is not None:
                prediction_dict[example_id] = prediction

        prediction_dump_path = self.prediction_dump_path + f"_{self.count}"
        with open(prediction_dump_path, "w") as f:
            json.dump(prediction_dict, f, indent=4)

        self.count += 1

        result_dict = evaluate_from_files(self.gold_data_path, prediction_dump_path)
        return result_dict[self.validation_metric]
