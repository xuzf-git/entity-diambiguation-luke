import torch
from allennlp.common import Registrable


class Retriever(Registrable):
    def __call__(self, scores: torch.Tensor) -> torch.LongTensor:
        """
        Parameters
        ----------
        scores: torch.Tensor (num_queries, num_targets)

        Returns
        -------
        target_indices: torch.LongTensor

        """
        raise NotImplementedError
