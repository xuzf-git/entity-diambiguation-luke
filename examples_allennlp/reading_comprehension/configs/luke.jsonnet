local pretrained_weight_path = std.extVar("PRETRAINED_WEIGHT_PATH");
local pretrained_metadata_path = std.extVar("PRETRAINED_METADATA_PATH");
local entity_vocab_path = std.extVar("ENTITY_VOCAB_PATH");

local base = import "lib/base.libsonnet";
local data = import "lib/data.libsonnet";

base + {
    "model": {
        "type": "transformers_qa",
        "embedder": {
            "type": "basic",
            "token_embedders": {
                "tokens": {
                    "type": "luke",
                    "pretrained_weight_path": pretrained_weight_path,
                    "pretrained_metadata_path": pretrained_metadata_path,
                    "entity_vocab_path": entity_vocab_path,
                    "discard_entity_embeddings": true,
                    "num_additional_special_tokens": std.length(data["extra_tokens"])
                }
            }
        },
    }
}

