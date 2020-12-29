import torch
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import TextFieldEmbedder
from allennlp.nn import util

from .seq2vec_encoder import Seq2VecEncoder


def masked_mean_pooling(embedding_sequence: torch.Tensor, mask: torch.Tensor):
    embedding_sequence = embedding_sequence * mask.unsqueeze(-1).float()

    summed_embeddings = embedding_sequence.sum(dim=1)  # shape: (batch_size, embedding_dim)

    lengths = mask.sum(dim=1)  # shape: (batch_size, )
    length_mask = lengths > 0

    # Set any length 0 to 1, to avoid dividing by zero.
    lengths = torch.max(lengths, lengths.new_ones(1))

    mean_pooled_embeddings = summed_embeddings / lengths.unsqueeze(-1).float()

    # mask embeddings with length 0
    mean_pooled_embeddings = mean_pooled_embeddings * (length_mask > 0).float().unsqueeze(-1)

    return mean_pooled_embeddings


@Seq2VecEncoder.register("boe")
class BoeEncoder(Seq2VecEncoder):
    def __init__(self, vocab: Vocabulary, embedder: TextFieldEmbedder, averaged: bool = False) -> None:
        super().__init__(vocab=vocab)
        self.embedder = embedder
        self.averaged = averaged

    def forward(self, tokens: TextFieldTensors) -> torch.Tensor:
        embedding_sequence = self.embedder(tokens)
        mask = util.get_text_field_mask(tokens)

        if self.averaged:
            return masked_mean_pooling(embedding_sequence, mask)
        else:
            embedding_sequence = embedding_sequence * mask.unsqueeze(-1).float()
            summed = embedding_sequence.sum(dim=1)  # shape: (batch_size, embedding_dim)
            return summed
