local pretrained_weight_path = std.extVar("PRETRAINED_WEIGHT_PATH");
local pretrained_metadata_path = std.extVar("PRETRAINED_METADATA_PATH");


local base = import "lib/base.libsonnet";

base + {
    "model": {
        "type": "dual_encoder_retrieval",
        "encoder": {
            "type": "first_token",
            "embedder": {
                "type": "basic",
                "token_embedders": {
                    "tokens": {
                        "type": "luke",
                        "pretrained_weight_path": pretrained_weight_path,
                        "pretrained_metadata_path": pretrained_metadata_path
                    }
                }
            },
        },
        "criterion": {"type": "in-batch_softmax"},
        "evaluate_top_k": 11,
        "normalize_embeddings": true
    }
}

