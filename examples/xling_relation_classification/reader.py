from typing import Dict
from allennlp.data import DatasetReader, TokenIndexer, Tokenizer, Instance
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.fields import SpanField, TextField, LabelField


def parse_relx_file(path: str):
    with open(path, "r") as f:
        for instance in f.read().strip().split("\n\n"):
            input_line, label = instance.strip().split("\n")
            example_id, input_sentence = input_line.split("\t")

            # make kbp37 data look like relx
            input_sentence = input_sentence.strip('"').strip().replace(" .", ".")
            yield {"example_id": example_id, "sentence": input_sentence, "label": label}


E1_START = "<e1>"
E1_END = "</e1>"
E2_START = "<e2>"
E2_END = "</e2>"


@DatasetReader.register("kbp37")
class KBP37Reader(DatasetReader):
    def __init__(self, tokenizer: Tokenizer, token_indexers: Dict[str, TokenIndexer], **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers

    def text_to_instance(self, sentence: str, label: str):
        tokens = self.tokenizer.tokenize(sentence)
        if isinstance(self.tokenizer, PretrainedTransformerTokenizer):
            tokens = self.tokenizer.add_special_tokens(tokens)
        text_field = TextField(tokens, token_indexers=self.token_indexers)

        texts = [t.text for t in tokens]

        e1_start_position = texts.index(E1_START)
        e1_end_position = texts.index(E1_END)
        e2_start_position = texts.index(E2_START)
        e2_end_position = texts.index(E2_END)

        return Instance(
            {
                "word_ids": text_field,
                "entity1_span": SpanField(e1_start_position, e1_end_position, text_field),
                "entity2_span": SpanField(e2_start_position, e2_end_position, text_field),
                "labels": LabelField(label),
            }
        )

    def _read(self, file_path: str):
        for data in parse_relx_file(file_path):
            yield self.text_to_instance(data["sentence"], data["label"])
