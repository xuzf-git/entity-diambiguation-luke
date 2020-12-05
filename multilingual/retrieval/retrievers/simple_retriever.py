import torch
from .retriever import Retriever


@Retriever.register("simple")
class SimpleRetriever(Retriever):
    """
    Simply retrieve the target with the highest score.
    """

    def __call__(self, scores: torch.Tensor):
        return scores.argmax(dim=1)
