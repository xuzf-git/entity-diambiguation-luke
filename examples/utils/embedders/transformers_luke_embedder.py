from typing import Optional
from overrides import overrides

import torch
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from transformers.models.luke.modeling_luke import LukeModel


@TokenEmbedder.register("transformers-luke")
class TransformersLukeEmbedder(TokenEmbedder):
    def __init__(
        self,
        model_name: str,
        train_parameters: bool = True,
        gradient_checkpointing: bool = False,
        output_entity_embeddings: bool = False,
    ) -> None:
        """

        Parameters
        ----------
        model_name: str
            Model name registered in transformers

        train_parameters: `bool`
            Decide if tunening or freezing pre-trained weights.

        gradient_checkpointing: `bool`
            Enable gradient checkpoinitng, which significantly reduce memory usage.
        output_entity_embeddings: `bool`
            If specified, the model returns entity embeddings instead of token embeddings.
            If you need both, please use PretrainedLukeEmbedderWithEntity.
        """
        super().__init__()

        self.output_entity_embeddings = output_entity_embeddings

        self.luke_model = LukeModel.from_pretrained(model_name)
        if not train_parameters:
            for param in self.transformer_model.parameters():
                param.requires_grad = False

    def get_output_dim(self):
        return self.luke_model.config.hidden_size

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

        if self.output_entity_embeddings:
            assert entity_ids is not None

        luke_outputs = self.luke_model(
            input_ids=token_ids,
            token_type_ids=type_ids,
            attention_mask=mask,
            entity_ids=entity_ids,
            entity_position_ids=entity_position_ids,
            entity_token_type_ids=entity_segment_ids,
            entity_attention_mask=entity_attention_mask,
        )

        if self.output_entity_embeddings:
            return luke_outputs.entity_last_hidden_state
        else:
            return luke_outputs.last_hidden_state
