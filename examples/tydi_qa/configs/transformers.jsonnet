local transformers_model_name = std.extVar("TRANSFORMERS_MODEL_NAME");

local base = import "lib/base.libsonnet";

base + {
    "model": {
        "type": "transformers_qa",
        "embedder": {
            "type": "basic",
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer",
                        "model_name": transformers_model_name
                    }
                }
            }
    }
}

