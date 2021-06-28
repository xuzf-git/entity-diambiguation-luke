from typing import List, Dict
import json
import numpy as np

from allennlp.data.fields import MetadataField, TensorField
from allennlp.data import DatasetReader, Instance, Token
from allennlp_models.rc.dataset_readers import TransformerSquadReader

from transformers.models.luke.tokenization_luke import LukeTokenizer


from examples.utils.wiki_mention_detector import WikiMentionDetector


@DatasetReader.register("transformers_squad")
class SquadReader(TransformerSquadReader):
    def __init__(self, transformer_model_name: str, mention_detector: WikiMentionDetector = None, **kwargs):
        super().__init__(transformer_model_name, **kwargs)
        if mention_detector is not None:
            mention_detector.set_tokenizer(self._tokenizer.tokenizer)

        self.mention_detector = mention_detector

    def _get_idx_to_title_mapping(self, file_path: str) -> Dict[str, str]:
        data = json.load(open(file_path, "r"))["data"]
        idx_to_title = {}
        for article in data:
            title = article["title"]
            for paragraph in article["paragraphs"]:
                for qa in paragraph["qas"]:
                    idx = qa["id"]
                    idx_to_title[idx] = title
        return idx_to_title

    def _read(self, file_path: str):

        idx_to_title_mapping = self._get_idx_to_title_mapping(file_path)

        for instance in super()._read(file_path):
            input_tokens = [t.text for t in instance["question_with_context"]]
            instance = Instance(
                {
                    "question_with_context": instance["question_with_context"],
                    "answer_span": instance["answer_span"],
                    "metadata": MetadataField(
                        {"input_tokens": input_tokens, "example_id": instance["metadata"].metadata["id"]}
                    ),
                }
            )

            if self.mention_detector is not None:
                index = instance["metadata"].metadata["example_id"]
                entity_features = self.get_entity_features(
                    instance["question_with_context"], title=idx_to_title_mapping[index]
                )
                for key, value in entity_features.items():
                    instance.add_field(key, value)

    def get_entity_features(self, tokens: List[Token], title: str):

        mentions = self.mention_detector.detect_mentions(tokens, title, "en")[: self.max_num_entity_features]

        entity_features = self.mention_detector.mentions_to_entity_features(tokens, mentions)

        entity_feature_fields = {}
        for name, feature in entity_features.items():
            entity_feature_fields[name] = TensorField(np.array(feature), padding_value=0)
        return entity_feature_fields
