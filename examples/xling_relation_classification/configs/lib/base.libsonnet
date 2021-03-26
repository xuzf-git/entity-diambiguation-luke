local bert_model_name = std.extVar("BERT_MODEL_NAME");
local train_data_path = std.extVar("TRAIN_DATA_PATH");
local validation_data_path = std.extVar("VALIDATION_DATA_PATH");
local test_data_path = std.extVar("TEST_DATA_PATH");

local batch_size = std.parseInt(std.extVar("BATCH_SIZE"));
local accumulation_steps = std.parseInt(std.extVar("ACCUMULATION_STEPS"));

local num_epochs = 20;
local num_steps_per_epoch = std.parseInt(std.extVar("NUM_STEPS_PER_EPOCH"));

local base = import "lib/base.libsonnet";

local tokenizer = {"type": "pretrained_transformer", "model_name": bert_model_name, "add_special_tokens": false};
local token_indexers = {
            "tokens": {"type": "pretrained_transformer", "model_name": bert_model_name}
    };


{
    "dataset_reader": {
        "type": "kbp37",
        "tokenizer": tokenizer,
        "token_indexers": token_indexers
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
            "lr": 2e-5,
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
            "warmup_steps": num_steps_per_epoch * num_epochs / 10
        },
        "num_gradient_accumulation_steps": accumulation_steps,
        "patience": 3,
        "validation_metric": "+f1"
    },
    "data_loader": {"batch_size": batch_size, "shuffle": true}
}