# Relation Classification
In this code, you can experiment with the task of relation classification with several datasets. Currently, we support the following datasets.


#####  English 
* [KBP37](https://arxiv.org/abs/1508.01006)
* [TACRED](https://www.aclweb.org/anthology/D17-1004/)

#####  Multilingual Evaluation
* [RELX](https://www.aclweb.org/anthology/2020.findings-emnlp.32/)

# Download datasets
```bash
cd data
git clone https://github.com/zhangdongxu/kbp37.git
git clone https://github.com/boun-tabi/RELX.git
```

For the TACRED dataset, you need to access through [LDC](https://catalog.ldc.upenn.edu/LDC2018T24).

# Training
We configure some parameters through environmental variables.  
Note that you need provide what dataset you are dealing with through `TASK`.
```bash
export TRANSFORMERS_MODEL_NAME="luke";
export TASK="kbp37"; # or "tacred".
export TRAIN_DATA_PATH="data/kbp37/train.txt";
export VALIDATION_DATA_PATH="data/kbp37/dev.txt";

# train LUKE
export TRANSFORMERS_MODEL_NAME="studio-ousia/luke-base";
poetry run allennlp train examples/relation_classification_allennlp/configs/transformers.jsonnet -s results/relation_classification/luke-base --include-package examples -o '{"trainer": {"cuda_device": 0}}'

# you can also fine-tune models from the BERT family
export TRANSFORMERS_MODEL_NAME="roberta-base";
poetry run allennlp train examples/relation_classification_allennlp/configs/transformers_luke.jsonnet  -s results/relation_classification/roberta-base --include-package examples
```

# Evaluation
```bash
poetry run allennlp evaluate RESULT_SAVE_DIR INPUT_FILE --include-package examples --output-file OUTPUT_FILE 

# example for LUKE
poetry run allennlp evaluate results/relation_classification/luke-base data/ultrafine_acl18/crowd/test.json --include-package examples --output-file results/relation_classification/luke-base/metrics_test.json --cuda 0
```

# Make Prediction
```bash
poetry run allennlp predict RESULT_SAVE_DIR INPUT_FILE --use-dataset-reader --include-package examples --cuda-device CUDA_DEVICE --output-file OUTPUT_FILE

# example for LUKE
poetry run allennlp predict results/relation_classification/luke-base data/ultrafine_acl18/crowd/dev.json --use-dataset-reader --include-package examples --cuda-device 0 --output-file results/relation_classification/luke-base/prediction.json
```

