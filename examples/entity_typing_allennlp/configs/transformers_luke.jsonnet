local transformers_model_name = std.extVar("TRANSFORMERS_MODEL_NAME");

local base = import "lib/base.libsonnet";

local extra_tokens = base["dataset_reader"]["tokenizer"]["tokenizer_kwargs"]["additional_special_tokens"];
base + {
    "model": {
        "type": "entity_typing",
        "feature_extractor": {
            "type": "entity",
            "embedder": {
                "type": "transformers-luke",
                "model_name": transformers_model_name,
                "output_entity_embeddings": true
            }
        },
    },
    "dataset_reader": base["dataset_reader"] + {"use_entity_feature": true},
}