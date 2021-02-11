from typing import Set, Dict, List, NamedTuple
from examples.reading_comprehension.utils.feature import convert_examples_to_features
from examples.reading_comprehension.utils.wiki_link_db import WikiLinkDB
import joblib
from transformers import AutoTokenizer
from luke.utils.entity_vocab import EntityVocab

from allennlp.data import Token


class Mention(NamedTuple):
    entity: str
    start: int
    end: int


class WikiMentionDetector:
    """
    Detect entity mentions in text from Wikipedia articles.
    """

    def __init__(
        self,
        wiki_link_db_path: str,
        model_redirect_mappings_path: str,
        link_redirect_mappings_path: str,
        min_mention_link_prob: float = 0.01,
        max_mention_length: int = 10
    ):
        self.wiki_link_db = WikiLinkDB(wiki_link_db_path)
        self.model_redirect_mappings = joblib.load(model_redirect_mappings_path)
        self.link_redirect_mappings = joblib.load(link_redirect_mappings_path)

        self.min_mention_link_prob = min_mention_link_prob

        self.max_mention_length = max_mention_length

    def get_mention_candidates(self, title: str) -> Dict[str, str]:
        title = self.link_redirect_mappings.get(title, title)

        if title not in self.wiki_link_db:
            return {}

        # mention_to_entity
        mention_candidates: Dict[str, str] = {}
        ambiguous_mentions: Set[str] = set()

        for link in self.wiki_link_db.get(title):
            if link.link_prob < self.min_mention_link_prob:
                continue

            link_text = self._normalize_mention(link.text)
            if link_text in mention_candidates and mention_candidates[link_text] != link.title:
                ambiguous_mentions.add(link_text)
                continue

            mention_candidates[link_text] = link.title

        for link_text in ambiguous_mentions:
            del mention_candidates[link_text]
        return mention_candidates

    def detect_mentions(self, tokens: List[str], mention_candidates: Dict[str, str]) -> List[Mention]:
        mentions = []
        cur = 0
        for start, token in enumerate(tokens):
            if start < cur:
                continue

            for end in range(min(start + self.max_mention_length, len(tokens)), start, -1):

                mention_text = self._tokenizer.convert_tokens_to_string(tokens[start:end])
                mention_text = self._normalize_mention(mention_text)
                if mention_text in mention_candidates:
                    cur = end
                    title = mention_candidates[mention_text]
                    title = self.model_redirect_mappings.get(title, title)  # resolve mismatch between two dumps
                    if title in self._entity_vocab:
                        mention = Mention(self._entity_vocab[title], start, end)
                        mentions.append(mention)
                    break

        return mentions

    def __call__(self, tokens: List[Token], title: str):
        pass

    @staticmethod
    def _normalize_mention(text: str):
        return " ".join(text.lower().split(" ")).strip()
