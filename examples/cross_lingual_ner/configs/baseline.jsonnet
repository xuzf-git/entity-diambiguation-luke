{
    "dataset_reader": {
        "type": "conll2003",
        "coding_scheme": "BIOUL",
        "tag_label": "ner",
        "token_indexers": {
            "token_characters": {
                "type": "characters",
                "min_padding_length": 3
            },
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            }
        }
    },
    "model": {
        "type": "crf_tagger",
        "encoder": {
            "type": "lstm",
            "bidirectional": true,
            "dropout": 0.5,
            "hidden_size": 200,
            "input_size": 428,
            "num_layers": 2
        },
        "label_encoding": "BIOUL",
        "constrain_crf_decoding": true,
        "regularizer": {
            "regexes": [
                [
                    "scalar_parameters",
                    {
                        "alpha": 0.1,
                        "type": "l2"
                    }
                ]
            ]
        },
        "text_field_embedder": {
            "token_embedders": {
                "token_characters": {
                    "type": "character_encoding",
                    "embedding": {
                        "embedding_dim": 16,
                        "vocab_namespace": "token_characters"
                    },
                    "encoder": {
                        "type": "cnn",
                        "conv_layer_activation": "relu",
                        "embedding_dim": 16,
                        "ngram_filter_sizes": [
                            3
                        ],
                        "num_filters": 128
                    }
                },
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 300,
                    "trainable": true
                }
            }
        }
    },
    "train_data_path": "data/ner_conll/en/train.txt",
    "validation_data_path": "data/ner_conll/en/valid.txt",
    "test_data_path": "data/ner_conll/en/test.txt",
    "trainer": {
        "cuda_device": -1,
        "grad_norm": 5,
        "num_epochs": 35,
        "optimizer": {
            "type": "adam",
            "lr": 0.001
        },
        "patience": 8,
        "validation_metric": "+f1-measure-overall"
    },
    "vocabulary": {"type": "from_instances"},
    "data_loader": {"batch_size": 32, "shuffle": true}
}