local base = import "lib/base.libsonnet";

local pretrained_weight_path = std.extVar("PRETRAINED_WEIGHT_PATH");
local pretrained_metadata_path = std.extVar("PRETRAINED_METADATA_PATH");
local entity_vocab_path = std.extVar("ENTITY_VOCAB_PATH");

base + {
    "model": {
        "type": "exhausitce_ner",
        "embedder": {
            "type": "basic",
            "token_embedders": {
                "tokens": {
                    "type": "luke",
                    "pretrained_weight_path": pretrained_weight_path,
                    "pretrained_metadata_path": pretrained_metadata_path,
                    "entity_vocab_path": entity_vocab_path,
                    "only_use_mask_embedding": true
                }
            }
        }
    }
}