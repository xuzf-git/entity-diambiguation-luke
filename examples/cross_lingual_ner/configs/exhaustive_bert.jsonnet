local bert_model_name = std.extVar("BERT_MODEL_NAME");

local base = import "lib/base.libsonnet";

base + {
    "model": {
        "type": "exhausitce_ner",
        "embedder": {
            "type": "basic",
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer",
                        "model_name": bert_model_name
                    }
                }
            },
    }
}