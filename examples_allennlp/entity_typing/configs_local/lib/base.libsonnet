local seed = std.parseInt(std.extVar("SEED"));
local transformers_model_name = std.extVar("TRANSFORMERS_MODEL_NAME");
local train_data_path = std.extVar("TRAIN_DATA_PATH");
local validation_data_path = std.extVar("VALIDATION_DATA_PATH");

local lr = 1e-5;
local batch_size = 8;
local accumulation_steps = 1;
local num_epochs = 5;
local effective_batch_size = batch_size * accumulation_steps;

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
        "num_epochs": num_epochs,
        "checkpointer": {
            "keep_most_recent_by_count": 0
        },
        "optimizer": {
            "type": "adamw",
            "lr": lr,
            "betas": [0.9, 0.98],
            "eps": 1e-6,
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
            "type": "custom_linear_with_warmup",
            "warmup_ratio": 0.06
        },
        "num_gradient_accumulation_steps": accumulation_steps,
        "patience": 3,
        "validation_metric": "+micro_fscore"
    },
    "data_loader": {"batch_size": batch_size, "shuffle": true},
    "random_seed": seed,
    "numpy_seed": seed,
    "pytorch_seed": seed
}