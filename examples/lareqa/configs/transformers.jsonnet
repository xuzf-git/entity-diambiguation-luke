local transformers_model_name = std.extVar("TRANSFORMERS_MODEL_NAME");

local base = import "lib/base.libsonnet";

local tokenizer = {"type": "pretrained_transformer", "model_name": transformers_model_name, "add_special_tokens": true};
local token_indexers = {
            "tokens": {"type": "pretrained_transformer", "model_name": transformers_model_name}
    };

base + {
    "model": {
        "type": "dual_encoder_retrieval",
        "encoder": {
            "type": "first_token",
            "embedder": {
                "type": "basic",
                "token_embedders": {
                    "tokens": {
                        "type": "pretrained_transformer",
                        "model_name": transformers_model_name
                    }
                }
            },
        },
        "criterion": {"type": "in-batch_softmax"},
        "evaluate_top_k": 11,
        "normalize_embeddings": true
    }
}

