local model_size = std.parseInt(std.extVar("model_size"));
local model_size = 300;
local pretrained_file = std.extVar("PRETRAINED_PATH");
local batch_size = std.parseInt(std.extVar("BATCH_SIZE"));

{
    "source_dataset_reader": {
       "type": "bucc",
       "tokenizer": {"type": "spacy"},
       "token_indexers": {"tokens": {"type": "single_id", "namespace": "source_tokens"}},
       "stop_word_language": "german",
       "max_instances": 100

    },
    "target_dataset_reader": {
       "type": "bucc",
       "tokenizer": {"type": "spacy"},
       "token_indexers": {"tokens": {"type": "single_id", "namespace": "target_tokens"}},
       "stop_word_language": "english",
       "max_instances": 100
    },
    "vocabulary": {
        "pretrained_files": {"source_tokens": "/Users/linghan/Downloads/wiki.multi.de.vec",
                             "target_tokens": "/Users/linghan/Downloads/wiki.multi.en.vec"},
        "only_include_pretrained_words": true
        },
    "data_loader": {
        "batch_sampler": {"type": "max_tokens_sampler", "max_tokens": batch_size, "padding_noise": 0.0}
    },
    "source_model": {
        "type": "boe",
        "embedder": {
            "type": "basic",
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": model_size,
                    "vocab_namespace": "source_tokens",
                    "pretrained_file": "/Users/linghan/Downloads/wiki.multi.de.vec"
                }
            }
        },
        "averaged": true
    },
    "target_model": {
        "type": "boe",
        "embedder": {
            "type": "basic",
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": model_size,
                    "vocab_namespace": "target_tokens",
                    "pretrained_file": "/Users/linghan/Downloads/wiki.multi.en.vec"
                }
            }
        },
         "averaged": true
    }
}
