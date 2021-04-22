from typing import NamedTuple, Dict, List
from collections import defaultdict
import json

import torch

from allennlp.training.metrics import Metric


class TyDiPrediction(NamedTuple):
    start_byte_offset: int
    end_byte_offset: int
    score: float


class TyDiMetric(Metric):
    def __init__(self, prediction_dump_path: str):
        super().__init__()
        self.document_predictions = defaultdict(list)
        self.passage_answer_candidates = {}
        self.prediction_dump_path = prediction_dump_path

    def __init__(self, output_dict: Dict[torch.Tensor], metadata_list: List[Dict]):

        prediction_score = (
            output_dict["span_start_prediction_score"] + output_dict["span_end_prediction_score"]
        ).tolist()
        for metadata, start, end, score in zip(
            metadata_list, output_dict["span_start_prediction"], output_dict["span_end_prediction"], prediction_score,
        ):
            token_to_contexts_byte_mapping = metadata["token_to_contexts_byte_mapping"]
            contexts_start_byte_offset = token_to_contexts_byte_mapping[start.item()][0]
            contexts_end_byte_offset = token_to_contexts_byte_mapping[end.item()][1]

            if contexts_start_byte_offset == -1 or contexts_end_byte_offset == -1:
                start_byte_offset = end_byte_offset = -1
            else:
                context_to_plaintext_offset = metadata["context_to_plaintext_offset"]
                start_byte_offset = context_to_plaintext_offset[contexts_start_byte_offset]
                end_byte_offset = context_to_plaintext_offset[contexts_end_byte_offset]

            self.document_predictions[metadata["example_id"]].append(
                TyDiPrediction(start_byte_offset, end_byte_offset, score)
            )
            self.passage_answer_candidates[metadata["example_id"]] = metadata["passage_answer_candidates"]

    @staticmethod
    def get_passage_idx(start_byte_offset: int, end_byte_offset: int, passage_answer_candidates: List[Dict[str, int]]):
        for passage_idx, candidates in enumerate(passage_answer_candidates):
            if (
                candidates["plaintext_start_byte"] <= start_byte_offset
                and end_byte_offset <= candidates["plaintext_end_byte"]
            ):
                return passage_idx
        return -1

    def get_metric(self, reset: bool):
        if not reset:
            return {}

        with open(self.prediction_dump_path, "w") as f:
            for exmaple_id, predictions in self.document_predictions.items():
                prediction = max(predictions, key=lambda x: x.score)

                passage_idx = self.get_passage_idx(
                    prediction.start_byte_offset, prediction.end_byte_offset, self.passage_answer_candidates[example_id]
                )

                json_prediction = {
                    "example_id": exmaple_id,
                    "passage_answer_index": passage_idx,
                    "passage_answer_score": prediction.score,
                    "minimal_answer": {
                        "start_byte_offset": prediction.start_byte_offset,
                        "end_byte_offset": prediction.end_byte_offset,
                    },
                    "minimal_answer_score": prediction.score,
                    "yes_no_answer": "NONE",
                }
                f.write(json.dumps(json_prediction) + "\n")
