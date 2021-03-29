local base = import "lib/base.libsonnet";

local pretrained_weight_path = std.extVar("PRETRAINED_WEIGHT_PATH");
local pretrained_metadata_path = std.extVar("PRETRAINED_METADATA_PATH");
local entity_vocab_path = std.extVar("ENTITY_VOCAB_PATH");

base + {
    "dataset_reader": base["dataset_reader"],
    "model": {
        "type": "transformers_relation_classifier",
        "embedder": {
            "type": "basic",
            "token_embedders": {
                "tokens": {
                    "type": "luke",
                    "pretrained_weight_path": pretrained_weight_path,
                    "pretrained_metadata_path": pretrained_metadata_path,
                    "entity_vocab_path": entity_vocab_path,
                    "num_additional_special_tokens": 4
                }
            }
        },
        "feature_type": "entity_start"
    }
}