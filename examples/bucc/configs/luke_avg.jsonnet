local batch_size = std.parseInt(std.extVar("BATCH_SIZE"));

local bert_model_name = std.extVar("BERT_MODEL_NAME");
local pretrained_weight_path = std.extVar("PRETRAINED_WEIGHT_PATH");
local pretrained_metadata_path = std.extVar("PRETRAINED_METADATA_PATH");

{
    "dataset_reader": {
       "type": "bucc",
       "tokenizer": {"type": "pretrained_transformer", "model_name": bert_model_name},
       "token_indexers": {"tokens": {"type": "pretrained_transformer",
                                     "model_name": bert_model_name}},
    },
    "vocabulary": {},
    "data_loader": {
        "batch_sampler": {"type": "max_tokens_sampler", "max_tokens": batch_size, "padding_noise": 0.0}
    },
    "model": {
        "type": "boe",
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
        "averaged": true
    }
}
