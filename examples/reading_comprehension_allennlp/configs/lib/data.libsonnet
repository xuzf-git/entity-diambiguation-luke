local task = std.extVar("TASK");
local transformers_model_name = std.extVar("TRANSFORMERS_MODEL_NAME");


local task_data =  {
    "tydiqa": {
       "dataset_reader": {
           "type": "tydiqa",
           "transformers_model_name": transformers_model_name,
           "is_evaluation": false,
           "include_unknowns_probability": 0.1
       },
       "validation_dataset_reader": {
            "type": "tydiqa",
            "transformers_model_name": transformers_model_name,
            "is_evaluation": true,
            "include_unknowns_probability": 0.0
       },
       "dataset_size": 8714624
    },
    "squad": {
        "dataset_reader": {
            "type": "transformers_squad",
            "transformer_model_name": transformers_model_name,
            "skip_impossible_questions": false
        },
        "validation_dataset_reader": {
            "type": "transformers_squad",
            "transformer_model_name": transformers_model_name,
            "skip_impossible_questions": false
        },
        "dataset_size": 8714624
    }
 };


{
    "dataset_reader": task_data[task]["dataset_reader"],
    "validation_dataset_reader": task_data[task]["validation_dataset_reader"],
    "dataset_size": task_data[task]["dataset_size"],
    "extra_tokens": ["[ContextId=%d]" % [i] for i in std.range(0, 44)]
}
