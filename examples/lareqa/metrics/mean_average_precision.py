from typing import List

import numpy as np
import torch
from allennlp.training.metrics import Metric
from collections import Counter

from examples.utils.retrieval.scoring_functions import ScoringFunction, CosineSimilarity


@Metric.register("retrieval_mAP")
class MeanAveragePrecision(Metric):
    def __init__(self, k: int = 20, scoring_function: ScoringFunction = CosineSimilarity()):
        self.k = k
        self.scoring_function = scoring_function

        self.query_embeddings = []
        self.target_embeddings = []
        self.query_ids = []
        self.target_ids = []

    def __call__(
        self,
        query_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        query_ids: List[str],
        target_ids: List[str],
    ):
        self.query_embeddings.append(query_embeddings)
        self.target_embeddings.append(target_embeddings)
        self.query_ids += query_ids
        self.target_ids += target_ids

    def reset(self):
        self.query_embeddings = []
        self.target_embeddings = []
        self.query_ids = []
        self.target_ids = []

    def get_metric(self, reset: bool):
        assert len(self.query_embeddings) == len(self.target_embeddings)
        if not reset or len(self.query_embeddings) == 0:
            return 0.0
        query_embeddings = torch.cat(self.query_embeddings, dim=0)
        target_embeddings = torch.cat(self.target_embeddings, dim=0)

        similarity_scores = self.scoring_function(query_embeddings, target_embeddings)
        if self.k is None:
            # when k is unspecified, consider all the target embeddings
            k = similarity_scores.size(0)
        else:
            k = min(similarity_scores.size(1), self.k)

        retrieved_top_k_indices = torch.argsort(similarity_scores, dim=1, descending=True)[:, :k]

        retrieved_top_k_ids: List[List[str]] = []
        for top_k_indices in retrieved_top_k_indices:
            top_k_ids = [self.target_ids[int(i)] for i in top_k_indices]
            retrieved_top_k_ids.append(top_k_ids)

        target_ids_counter = Counter(self.target_ids)
        scores_for_each_query: List[float] = []
        for q_id, retrieved_ids in zip(self.query_ids, retrieved_top_k_ids):
            score = 0.0
            num_correct = 0
            for num_total, retrieved_id in enumerate(retrieved_ids, 1):
                is_correct = int(q_id == retrieved_id)
                num_correct += is_correct
                score += (num_correct / num_total) * is_correct
            score = score / target_ids_counter[q_id]
            scores_for_each_query.append(score)

        overall_score = np.mean(scores_for_each_query)
        if reset:
            self.reset()
        return overall_score
