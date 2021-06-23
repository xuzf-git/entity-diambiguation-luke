local transformers_model_name = std.extVar("TRANSFORMERS_MODEL_NAME");

local base = import "lib/base.libsonnet";

local extra_tokens = base["dataset_reader"]["tokenizer"]["tokenizer_kwargs"]["additional_special_tokens"];
base + {
    "model": {
        "type": "entity_typing",
        "feature_extractor": {
            "type": "token",
            "embedder": {
                "type": "pretrained_transformer",
                "model_name": transformers_model_name,
                "tokenizer_kwargs": {"additional_special_tokens": extra_tokens}
            },
            "feature_type": "entity_start"
        }
    }
}