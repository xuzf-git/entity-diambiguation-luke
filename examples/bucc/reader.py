from typing import Dict
from allennlp.data import DatasetReader, Instance, Tokenizer, TokenIndexer
from allennlp.data.fields import MetadataField, TextField


def parse_bucc_file(file_path: str):
    with open(file_path, "r") as f:
        for line in f:
            idx, sentence_or_idx = line.strip().split("\t")
            yield idx, sentence_or_idx


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

    def text_to_instance(self, sentence: str, idx: str) -> Instance:
        tokens = self.tokenizer.tokenize(sentence)
        if self.stop_words:
            tokens = [t for t in tokens if t.text not in self.stop_words]
        fields = {
            "tokens": TextField(tokens, self.token_indexers),
            "index": MetadataField(idx),
        }
        return Instance(fields)

    def _read(self, file_path: str):
        for idx, sentence in parse_bucc_file(file_path):
            yield self.text_to_instance(sentence, idx)
