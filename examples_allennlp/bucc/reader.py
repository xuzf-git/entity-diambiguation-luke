from typing import Dict, List, Tuple

from allennlp.data import DatasetReader, Instance, Tokenizer, TokenIndexer
from allennlp.data.fields import MetadataField, TextField


def parse_bucc_file(file_path: str):
    with open(file_path, "r") as f:
        for line in f:
            idx, sentence_or_idx = line.strip().split("\t")
            yield idx, sentence_or_idx


def read_gold_sentence_pairs(
    source_sentence_file: str, target_sentence_file: str, gold_file: str
) -> List[Tuple[str, str]]:
    source_id2sentence = {idx: sentence for idx, sentence in parse_bucc_file(source_sentence_file)}
    target_id2sentence = {idx: sentence for idx, sentence in parse_bucc_file(target_sentence_file)}

    gold_sentence_pairs = []
    for source_idx, target_idx in parse_bucc_file(gold_file):
        source_sentence = source_id2sentence[source_idx]
        target_sentence = target_id2sentence[target_idx]
        gold_sentence_pairs.append((source_sentence, target_sentence))
    return gold_sentence_pairs


@DatasetReader.register("bucc")
class BUCCReader(DatasetReader):
    def __init__(
        self, tokenizer: Tokenizer, token_indexers: Dict[str, TokenIndexer], stop_word_language: str = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers

        self.stop_word_language = stop_word_language
        if stop_word_language is not None:
            from nltk.corpus import stopwords

            self.stop_words = set(stopwords.words(stop_word_language))
        else:
            self.stop_words = set()

    def text_to_instance(self, sentence: str) -> Instance:
        tokens = self.tokenizer.tokenize(sentence)
        if self.stop_words:
            tokens = [t for t in tokens if t.text not in self.stop_words]
        fields = {
            "tokens": TextField(tokens, self.token_indexers),
            "sentence": MetadataField(sentence),
        }
        return Instance(fields)

    def _read(self, file_path: str):
        seen_sentences = set()
        for idx, sentence in parse_bucc_file(file_path):
            if sentence in seen_sentences:
                continue
            yield self.text_to_instance(sentence)
            seen_sentences.add(sentence)
