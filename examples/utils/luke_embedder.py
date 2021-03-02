from typing import Optional
import json
from overrides import overrides

import torch
from transformers import AutoConfig

from allennlp.modules.token_embedders.token_embedder import TokenEmbedder

from luke.model import LukeModel, LukeConfig


@TokenEmbedder.register("luke")
class PretrainedLukeEmbedder(TokenEmbedder):
    def __init__(
        self, pretrained_weight_path: str, pretrained_metadata_path: str, train_parameters: bool = True
    ) -> None:
        super().__init__()

        self.metadata = json.load(open(pretrained_metadata_path, "r"))["model_config"]

        config = LukeConfig(
            entity_vocab_size=self.metadata["entity_vocab_size"],
            bert_model_name=self.metadata["bert_model_name"],
            entity_emb_size=self.metadata["entity_emb_size"],
            **AutoConfig.from_pretrained(self.metadata["bert_model_name"]).to_dict(),
        )

        self.luke_model = LukeModel(config)
        self.luke_model.load_state_dict(torch.load(pretrained_weight_path), strict=False)

        if not train_parameters:
            for param in self.transformer_model.parameters():
                param.requires_grad = False

    @overrides
    def get_output_dim(self):
        return self.metadata["hidden_size"]

    @overrides
    def forward(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor] = None,
        entity_ids: torch.LongTensor = None,
        entity_position_ids: torch.LongTensor = None,
        entity_segment_ids: torch.LongTensor = None,
        entity_attention_mask: torch.LongTensor = None,
    ) -> torch.Tensor:  # type: ignore

        sequence_output = self.luke_model(
            token_ids,
            word_segment_ids=type_ids,
            word_attention_mask=mask,
            entity_ids=entity_ids,
            entity_position_ids=entity_position_ids,
            entity_segment_ids=entity_segment_ids,
            entity_attention_mask=entity_attention_mask,
        )[0]

        return sequence_output
