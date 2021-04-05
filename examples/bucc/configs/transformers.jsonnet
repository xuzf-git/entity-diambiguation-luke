local batch_size = std.parseInt(std.extVar("BATCH_SIZE"));
local model_name = std.extVar("TRANSFORMERS_MODEL_NAME");

local layer_index = std.parseInt(std.extVar("LAYER_INDEX"));

local last_layer_embedder = {
                    "type": "pretrained_transformer",
                    "model_name": model_name
};

local intermediate_embedder = {
                    "type": "intermediate_pretrained_transformer",
                    "model_name": model_name,
                    "layer_index": layer_index
};

local embedder = if layer_index == -1
    then last_layer_embedder
    else intermediate_embedder;

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
        "type": "boe",
        "embedder": {
            "type": "basic",
            "token_embedders": {
                "tokens": embedder
            }
        },
        "averaged": true
    }
}
