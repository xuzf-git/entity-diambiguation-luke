from typing import Optional
import json
from overrides import overrides

import torch
from transformers import AutoConfig

from allennlp.modules.token_embedders.token_embedder import TokenEmbedder

from luke.utils.entity_vocab import EntityVocab, MASK_TOKEN, PAD_TOKEN
from luke.model import LukeModel, LukeConfig


@TokenEmbedder.register("luke")
class PretrainedLukeEmbedder(TokenEmbedder):
    def __init__(
        self,
        pretrained_weight_path: str,
        pretrained_metadata_path: str,
        entity_vocab_path: str = None,
        train_parameters: bool = True,
        gradient_checkpointing: bool = False,
        only_use_mask_embedding: bool = False,
    ) -> None:
        super().__init__()

        self.metadata = json.load(open(pretrained_metadata_path, "r"))["model_config"]
        if entity_vocab_path is not None:
            self.entity_vocab = EntityVocab(entity_vocab_path)
        else:
            self.entity_vocab = None

        model_weights = torch.load(pretrained_weight_path, map_location=torch.device("cpu"))
        self.only_use_mask_embedding = only_use_mask_embedding
        if only_use_mask_embedding:
            assert self.entity_vocab is not None
            pad_id = self.entity_vocab.special_token_ids[PAD_TOKEN]
            mask_id = self.entity_vocab.special_token_ids[MASK_TOKEN]
            self.metadata["entity_vocab_size"] = 2
            entity_emb = model_weights["entity_embeddings.entity_embeddings.weight"]
            mask_emb = entity_emb[mask_id].unsqueeze(0)
            pad_emb = entity_emb[pad_id].unsqueeze(0)
            model_weights["entity_embeddings.entity_embeddings.weight"] = torch.cat([pad_emb, mask_emb])

        config = LukeConfig(
            entity_vocab_size=self.metadata["entity_vocab_size"],
            bert_model_name=self.metadata["bert_model_name"],
            entity_emb_size=self.metadata["entity_emb_size"],
            **AutoConfig.from_pretrained(self.metadata["bert_model_name"]).to_dict(),
        )
        config.gradient_checkpointing = gradient_checkpointing

        self.luke_model = LukeModel(config)
        self.luke_model.load_state_dict(model_weights, strict=False)

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
