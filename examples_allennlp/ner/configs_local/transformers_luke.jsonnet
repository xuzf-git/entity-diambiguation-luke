local transformers_model_name = std.extVar("TRANSFORMERS_MODEL_NAME");

local base = import "lib/base.libsonnet";

base + {
    "dataset_reader": base["dataset_reader"] + {"use_entity_feature": true},
    "model": {
        "type": "exhaustive_ner",
        "feature_extractor": {
            "type": "token-entity",
            "embedder": {
                "type": "transformers-luke",
                "model_name": transformers_model_name,
                "output_embeddings": "tokens+entity"
            }
        }
    }
}