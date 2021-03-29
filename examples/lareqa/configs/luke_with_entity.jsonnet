local pretrained_weight_path = std.extVar("PRETRAINED_WEIGHT_PATH");
local pretrained_metadata_path = std.extVar("PRETRAINED_METADATA_PATH");


local base = import "lib/base.libsonnet";


local mention_detector = {
        "wiki_link_db_path": "/home/ryo0123/wiki_link_data/enwiki-20160501-wiki-link.db",
        "model_redirect_mappings_path": "/home/ryo0123/wiki_link_data/enwiki-20160501-redirect.pkl",
        "link_redirect_mappings_path": "/home/ryo0123/wiki_link_data/enwiki-20160501-redirect.pkl",
        "inter_wiki_path": "/home/ryo0123/wiki_link_data/interwiki-20160502-all.db",
        "entity_vocab_path": "/data/luke/xlm_roberta_base/entity_vocab.jsonl",
};

local multilingual_mention_detector = {
        "wiki_link_db_path": "/home/ryo0123/wiki_link_data/enwiki-20160501-wiki-link.db",
        "model_redirect_mappings_path": "/home/ryo0123/wiki_link_data/enwiki-20160501-redirect.pkl",
        "link_redirect_mappings_path": "/home/ryo0123/wiki_link_data/enwiki-20160501-redirect.pkl",
        "inter_wiki_path": "/home/ryo0123/wiki_link_data/interwiki-20160502-all.db",
        "entity_vocab_path": "/data/luke/xlm_roberta_base/entity_vocab.jsonl",
        "multilingual_entity_db_path": {
            "ar": "/home/ryo0123/wiki_link_data/entity_db/arwiki-20201201-entity.db",
            "de": "/home/ryo0123/wiki_link_data/entity_db/dewiki-20201201-entity.db",
            "el": "/home/ryo0123/wiki_link_data/entity_db/elwiki-20201201-entity.db",
            "es": "/home/ryo0123/wiki_link_data/entity_db/eswiki-20201201-entity.db",
            "hi": "/home/ryo0123/wiki_link_data/entity_db/hiwiki-20201201-entity.db",
            "ru": "/home/ryo0123/wiki_link_data/entity_db/ruwiki-20201201-entity.db",
            "th": "/home/ryo0123/wiki_link_data/entity_db/thwiki-20201201-entity.db",
            "tr": "/home/ryo0123/wiki_link_data/entity_db/trwiki-20201201-entity.db",
            "vi": "/home/ryo0123/wiki_link_data/entity_db/viwiki-20201201-entity.db",
            "zh": "/home/ryo0123/wiki_link_data/entity_db/zhwiki-20201201-entity.db"

        }
};

base + {
    "dataset_reader": base["dataset_reader"] + {"wiki_mention_detector": mention_detector},
    "validation_dataset_reader": base["validation_dataset_reader"] + {"wiki_mention_detector": multilingual_mention_detector},
    "model": {
        "type": "dual_encoder_retrieval",
        "encoder": {
            "type": "first_token",
            "embedder": {
                "type": "basic",
                "token_embedders": {
                    "tokens": {
                        "type": "luke",
                        "pretrained_weight_path": pretrained_weight_path,
                        "pretrained_metadata_path": pretrained_metadata_path
                    }
                }
            },
        },
        "criterion": {"type": "in-batch_softmax"},
        "evaluate_top_k": 11,
        "normalize_embeddings": true
    }
}

