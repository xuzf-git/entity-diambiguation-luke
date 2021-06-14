local pretrained_weight_path = std.extVar("PRETRAINED_WEIGHT_PATH");
local pretrained_metadata_path = std.extVar("PRETRAINED_METADATA_PATH");
local entity_vocab_path = std.extVar("ENTITY_VOCAB_PATH");

local base = import "lib/base.libsonnet";
local data = import "lib/data.libsonnet";

local languages = ["ar", "bn", "en", "fi", "id", "ja", "ko", "ru", "te", "sw", "th"];
local mention_detectors = [
{"wiki_link_db_path": "/home/ryo0123/tydiqa_wiki_link_data/wikilink_db/%swiki-20190201-wikilink.db" % [l],
 "model_redirect_mappings_path": "/home/ryo0123/tydiqa_wiki_link_data/redirect_files/%swiki-20190201-redirect.pkl" % [l],
 "link_redirect_mappings_path": "/home/ryo0123/tydiqa_wiki_link_data/redirect_files/%swiki-20190201-redirect.pkl" % [l],
 "entity_vocab_path": "/data/luke/xlm_roberta_base/entity_vocab.jsonl",
 "source_language": l} for l in languages
 ];

base + {
    "dataset_reader": base["dataset_reader"] + {"mention_detectors": mention_detectors},
    "validation_dataset_reader": base["validation_dataset_reader"] + {"mention_detectors": mention_detectors},
    "model": {
        "type": "transformers_qa",
        "embedder": {
            "type": "basic",
            "token_embedders": {
                "tokens": {
                    "type": "luke",
                    "pretrained_weight_path": pretrained_weight_path,
                    "pretrained_metadata_path": pretrained_metadata_path,
                    "entity_vocab_path": entity_vocab_path,
                    "num_additional_special_tokens": std.length(data["extra_tokens"])
                }
            }
        },
    }
}

