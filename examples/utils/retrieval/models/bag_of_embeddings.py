import torch
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import TextFieldEmbedder
from allennlp.nn import util

from .seq2vec_encoder import Seq2VecEncoder


@Seq2VecEncoder.register("boe")
class BoeEncoder(Seq2VecEncoder):
    def __init__(self, vocab: Vocabulary, embedder: TextFieldEmbedder, averaged: bool = False) -> None:
        super().__init__(vocab=vocab)
        self.embedder = embedder
        self.averaged = averaged

    def get_input_dim(self):
        return self.embedding_dim

    def get_output_dim(self):
        return self.embedding_dim

    def forward(self, tokens: TextFieldTensors) -> torch.Tensor:
        embeddings = self.embedder(tokens)
        mask = util.get_text_field_mask(tokens)

        embeddings = embeddings * mask.unsqueeze(-1).float()

        summed = embeddings.sum(dim=1)  # shape: (batch_size, embedding_dim)

        if self.averaged:
            lengths = mask.sum(dim=1)  # shape: (batch_size, )
            length_mask = lengths > 0

            # Set any length 0 to 1, to avoid dividing by zero.
            lengths = torch.max(lengths, lengths.new_ones(1))

            summed = summed / lengths.unsqueeze(-1).float()

            if length_mask is not None:
                summed = summed * (length_mask > 0).float().unsqueeze(-1)

        return summed
