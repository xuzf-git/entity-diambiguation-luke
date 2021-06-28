local transformers_model_name = std.extVar("TRANSFORMERS_MODEL_NAME");

local base = import "lib/base.libsonnet";
local data = import "lib/data.libsonnet";

local mention_detector =  {
    "wiki_link_db_path": std.extVar("WIKI_LINK_DB_PATH"),
    "model_redirect_mappings_path": std.extVar("MODEL_REDIRECT_MAPPINGS_PATH"),
    "link_redirect_mappings_path": std.extVar("LINK_REDIRECT_MAPPINGS_PATH"),
    "entity_vocab_path": transformers_model_name
};

base + {
    "dataset_reader": base["dataset_reader"] + {"mention_detector": mention_detector},
    "validation_dataset_reader": base["validation_dataset_reader"] + {"mention_detector": mention_detector},
    "model": {
        "type": "transformers_qa",
        "embedder": {
            "type": "basic",
            "token_embedders": {
                "tokens": {
                    "type": "transformers-luke",
                    "model_name": transformers_model_name,
                    }
                }
            }
    }
}

