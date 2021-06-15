local transformers_model_name = std.extVar("TRANSFORMERS_MODEL_NAME");
local train_data_path = std.extVar("TRAIN_DATA_PATH");
local validation_data_path = std.extVar("VALIDATION_DATA_PATH");

local batch_size = 4;
local accumulation_steps = 1;
local effective_batch_size = batch_size * accumulation_steps;
local num_epochs = 3;

local base = import "lib/base.libsonnet";

local dataset_size = 2000;


local extra_tokens = ["<ent>"];

local tokenizer = {"type": "pretrained_transformer",
                   "model_name": transformers_model_name,
                   "add_special_tokens": true,
                   "tokenizer_kwargs": {"additional_special_tokens": extra_tokens}};
local token_indexers = {
            "tokens": {"type": "pretrained_transformer", "model_name": transformers_model_name,
                       "tokenizer_kwargs": {"additional_special_tokens": extra_tokens}
    }};

{
    "dataset_reader": {
        "type": "entity_typing",
        "tokenizer": tokenizer,
        "token_indexers": token_indexers
    },
    "train_data_path": train_data_path,
    "validation_data_path": validation_data_path,
    "trainer": {
        "cuda_device": -1,
        "grad_norm": 5,
        "num_epochs": 5,
        "checkpointer": {
            "keep_most_recent_by_count": 0
        },
        "optimizer": {
            "type": "adamw",
            "lr": 1e-5,
            "weight_decay": 0.01,
            "parameter_groups": [
                [
                    [
                        "bias",
                        "LayerNorm.weight",
                    ],
                    {
                        "weight_decay": 0
                    }
                ]
            ],
        },
        "learning_rate_scheduler": {
            "type": "linear_with_warmup",
            "warmup_steps": std.floor((dataset_size / effective_batch_size) * num_epochs / 10)
        },
        "num_gradient_accumulation_steps": accumulation_steps,
        "patience": 2,
        "validation_metric": "+micro_fscore"
    },
    "data_loader": {"batch_size": batch_size, "shuffle": true}
}