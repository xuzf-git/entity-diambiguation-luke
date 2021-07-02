local seed = std.parseInt(std.extVar("SEED"));
local train_data_path = std.extVar("TRAIN_DATA_PATH");
local validation_data_path = std.extVar("VALIDATION_DATA_PATH");

local lr =  1.5e-5;
local batch_size = 3;
local accumulation_steps = 16;
local num_epochs = 2;
local effective_batch_size = batch_size * accumulation_steps;

local data = import "data.libsonnet";

{
    "dataset_reader": data["dataset_reader"],
    "validation_dataset_reader": data["validation_dataset_reader"],
    "train_data_path": train_data_path,
    "validation_data_path": validation_data_path,
    "data_loader": {
        "batch_size": batch_size, "shuffle": true,
        "num_workers": 0,
        "max_instances_in_memory": 50000
    },
     "validation_data_loader": {
        "batch_size": batch_size, "shuffle": false, "num_workers": 0,
        "max_instances_in_memory": 5000
    },
    "trainer": {
        "num_epochs": num_epochs,
        "patience": 3,
        "cuda_device": -1,
        "grad_norm": 5.0,
        "num_gradient_accumulation_steps": accumulation_steps,
        "checkpointer": {
            "keep_most_recent_by_count": 0
        },
        "validation_metric": "-loss",
        "optimizer": {
            "type": "adamw",
            "lr": lr,
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
            "warmup_steps": std.floor((data["dataset_size"] / effective_batch_size) * num_epochs * 0.06),
            "num_epochs": num_epochs,
            "num_steps_per_epoch": data["dataset_size"] / effective_batch_size
        },
    },
    "random_seed": seed,
    "numpy_seed": seed,
    "pytorch_seed": seed
}
