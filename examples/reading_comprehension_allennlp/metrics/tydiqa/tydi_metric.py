from typing import NamedTuple, Dict, List
from collections import defaultdict
import json
from pathlib import Path

import torch

from examples.reading_comprehension_allennlp.metrics.tydiqa.tydi_eval import evaluate_prediction_file
from examples.reading_comprehension_allennlp.metrics.qa_metric import QAMetric


class TyDiPrediction(NamedTuple):
    start_byte_offset: int
    end_byte_offset: int
    score: float


@QAMetric.register("tydi-qa")
class TyDiMetric(QAMetric):
    def __init__(
        self,
        gold_data_path: str,
        prediction_dump_path: str,
        validation_language: str = "average",
        validation_task: str = "minimal_answer",
        validation_metric: str = "f1",
    ):
        super().__init__()
        self.document_predictions = defaultdict(list)
        self.passage_answer_candidates = {}
        self.example_to_language = {}
        assert Path(gold_data_path).exists()
        self.gold_data_path = gold_data_path
        self.prediction_dump_path = prediction_dump_path

        self.validation_language = validation_language
        self.validation_task = validation_task
        self.validation_metric = validation_metric

        self.count = 0

    @property
    def validation_metric_name(self) -> str:
        return f"{self.validation_language}_{self.validation_task}_{self.validation_metric}"

    def __call__(self, output_dict: Dict[str, torch.Tensor], metadata_list: List[Dict]):

        prediction_score = output_dict["prediction_score"].tolist()
        for metadata, (start, end), score in zip(metadata_list, output_dict["span_prediction"], prediction_score,):
            token_to_contexts_byte_mapping = metadata["token_to_contexts_byte_mapping"]
            contexts_start_byte_offset = token_to_contexts_byte_mapping[start.item()][0]
            contexts_end_byte_offset = token_to_contexts_byte_mapping[end.item()][1]

            if contexts_start_byte_offset == -1 or contexts_end_byte_offset == -1:
                start_byte_offset = end_byte_offset = -1
            else:
                context_to_plaintext_offset = metadata["context_to_plaintext_offset"]
                try:
                    start_byte_offset = context_to_plaintext_offset[contexts_start_byte_offset]
                    end_byte_offset = context_to_plaintext_offset[contexts_end_byte_offset]
                except IndexError:
                    import warnings

                    example_id = metadata["example_id"]
                    warnings.warn(f"IndexError happens with {example_id}")
                    start_byte_offset = end_byte_offset = -1

            # sanitize
            if start_byte_offset == -1 or end_byte_offset == -1 or start_byte_offset > end_byte_offset:
                start_byte_offset = end_byte_offset = -1

            self.document_predictions[metadata["example_id"]].append(
                TyDiPrediction(start_byte_offset, end_byte_offset, score)
            )
            self.passage_answer_candidates[metadata["example_id"]] = metadata["passage_answer_candidates"]
            self.example_to_language[metadata["example_id"]] = metadata["language"]

    @staticmethod
    def get_passage_idx(start_byte_offset: int, end_byte_offset: int, passage_answer_candidates: List[Dict[str, int]]):
        for passage_idx, candidates in enumerate(passage_answer_candidates):
            if (
                candidates["plaintext_start_byte"] <= start_byte_offset
                and end_byte_offset <= candidates["plaintext_end_byte"]
            ):
                return passage_idx
        return -1

    def get_metric(self, reset: bool) -> float:
        if not reset:
            return 0

        prediction_dump_path = self.prediction_dump_path + f"_{self.count}"
        with open(prediction_dump_path, "w") as f:
            for example_id, predictions in self.document_predictions.items():
                prediction = max(predictions, key=lambda x: x.score)

                passage_idx = self.get_passage_idx(
                    prediction.start_byte_offset, prediction.end_byte_offset, self.passage_answer_candidates[example_id]
                )

                json_prediction = {
                    "example_id": example_id,
                    "passage_answer_index": passage_idx,
                    "passage_answer_score": prediction.score,
                    "minimal_answer": {
                        "start_byte_offset": prediction.start_byte_offset,
                        "end_byte_offset": prediction.end_byte_offset,
                    },
                    "minimal_answer_score": prediction.score,
                    "yes_no_answer": "NONE",
                    "language": self.example_to_language[example_id],
                }
                f.write(json.dumps(json_prediction) + "\n")

                self.count += 1

        result_dict = evaluate_prediction_file(self.gold_data_path, prediction_dump_path)
        return result_dict[self.validation_language][self.validation_task][self.validation_metric]
