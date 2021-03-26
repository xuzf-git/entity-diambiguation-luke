local bert_model_name = std.extVar("BERT_MODEL_NAME");
local train_data_path = std.extVar("TRAIN_DATA_PATH");
local validation_data_path = std.extVar("VALIDATION_DATA_PATH");
local test_data_path = std.extVar("TEST_DATA_PATH");

local batch_size = std.parseInt(std.extVar("BATCH_SIZE"));
local accumulation_steps = std.parseInt(std.extVar("ACCUMULATION_STEPS"));

local base = import "lib/base.libsonnet";

local tokenizer = {"type": "pretrained_transformer", "model_name": bert_model_name, "add_special_tokens": false};
local token_indexers = {
            "tokens": {"type": "pretrained_transformer", "model_name": bert_model_name}
    };


{
    "dataset_reader": {
        "type": "conll_exhaustive",
        "tokenizer": tokenizer,
        "token_indexers": token_indexers,
        "encoding": "utf-8",
    },
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
    },
    "train_data_path": train_data_path,
    "validation_data_path": validation_data_path,
    "test_data_path": test_data_path,
    "trainer": {
        "cuda_device": -1,
        "grad_norm": 5,
        "num_epochs": 20,
        "optimizer": {
            "type": "adamw",
            "lr": 2e-5
        },
        "num_gradient_accumulation_steps": accumulation_steps,
        "patience": 3,
        "validation_metric": "+f1"
    },
    "data_loader": {"batch_size": batch_size, "shuffle": true}
}