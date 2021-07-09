local transformers_model_name = std.extVar("TRANSFORMERS_MODEL_NAME");

local base = import "lib/base.libsonnet";

base + {
    "model": {
        "type": "exhaustive_ner",
        "feature_extractor": {
            "type": "token",
            "embedder": {
                "type": "pretrained_transformer",
                "model_name": transformers_model_name
            }
        }
    }
}