local bert_model_name = std.extVar("BERT_MODEL_NAME");
local pretrained_weight_path = std.extVar("PRETRAINED_WEIGHT_PATH");
local pretrained_metadata_path = std.extVar("PRETRAINED_METADATA_PATH");


local base = import "lib/base.libsonnet";

local tokenizer = {"type": "pretrained_transformer", "model_name": bert_model_name, "add_special_tokens": true};
local token_indexers = {
            "tokens": {"type": "pretrained_transformer", "model_name": bert_model_name}
    };

base + {
    "dataset_reader": {
       "type": "lareqa",
       "mode": "squad",
       "tokenizer": tokenizer,
       "token_indexers": token_indexers},
    "validation_dataset_reader": {
       "type": "lareqa",
       "mode": "lareqa",
       "tokenizer": tokenizer,
       "token_indexers": token_indexers},
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

