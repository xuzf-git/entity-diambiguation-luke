local model_size = std.parseInt(std.extVar("model_size"));
local batch_size = std.parseInt(std.extVar("BATCH_SIZE"));

local model_name = "xlm-roberta-base";
{
    "dataset_reader": {
       "type": "bucc",
       "tokenizer": {"type": "pretrained_transformer", "model_name": model_name},
       "token_indexers": {"tokens": {"type": "pretrained_transformer",
                                     "model_name": model_name}},
    },
    "vocabulary": {},
    "data_loader": {
        "batch_sampler": {"type": "max_tokens_sampler", "max_tokens": batch_size, "padding_noise": 0.0}
    },
    "model": {
        "type": "first_token",
        "embedder": {
            "type": "basic",
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer",
                    "model_name": model_name
                }
            }
        },
    }
}
