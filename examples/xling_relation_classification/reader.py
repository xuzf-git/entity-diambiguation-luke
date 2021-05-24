from typing import Dict
import json
from pathlib import Path
import numpy as np
from allennlp.data import DatasetReader, TokenIndexer, Tokenizer, Instance
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.fields import SpanField, TextField, LabelField, ArrayField, MetadataField


E1_START = "<e1>"
E1_END = "</e1>"
E2_START = "<e2>"
E2_END = "</e2>"


def parse_kbp37_or_relx_file(path: str):
    with open(path, "r") as f:
        for instance in f.read().strip().split("\n\n"):
            input_line, label = instance.strip().split("\n")
            example_id, input_sentence = input_line.split("\t")

            # make kbp37 data look like relx
            input_sentence = input_sentence.strip('"').strip().replace(" .", ".")
            yield {"example_id": example_id, "sentence": input_sentence, "label": label}


def parse_tacred_file(path: str):
    if Path(path).suffix != ".json":
        raise ValueError(f"{path} does not seem to be a json file. We currently only supports the json format file.")
    for example in json.load(open(path, "r")):
        tokens = example["token"]
        insert_index = {
            example["subj_start"]: E1_START,
            example["subj_end"] + 1: E1_END,
            example["obj_start"]: E2_START,
            example["obj_end"] + 1: E2_END,
        }
        for i, token in sorted(insert_index.items(), key=lambda x: -x[0]):
            tokens.insert(i, token)

        yield {"example_id": example["id"], "sentence": " ".join(tokens), "label": example["relation"]}


@DatasetReader.register("relation_classification")
class RelationClassificationReader(DatasetReader):
    def __init__(
        self,
        dataset: str,
        tokenizer: Tokenizer,
        token_indexers: Dict[str, TokenIndexer],
        use_entity_feature: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if dataset == "kbp37":
            self.parser = parse_kbp37_or_relx_file
        elif dataset == "tacred":
            self.parser = parse_tacred_file
        else:
            raise ValueError(f"Valid values: [kbp37, tacred], but we got {dataset}")

        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.use_entity_feature = use_entity_feature

    def text_to_instance(self, sentence: str, label: str = None):
        tokens = self.tokenizer.tokenize(sentence)
        if isinstance(self.tokenizer, PretrainedTransformerTokenizer):
            tokens = self.tokenizer.add_special_tokens(tokens)
        text_field = TextField(tokens, token_indexers=self.token_indexers)

        texts = [t.text for t in tokens]

        e1_start_position = texts.index(E1_START)
        e1_end_position = texts.index(E1_END)
        e2_start_position = texts.index(E2_START)
        e2_end_position = texts.index(E2_END)

        fields = {
            "word_ids": text_field,
            "entity1_span": SpanField(e1_start_position, e1_end_position, text_field),
            "entity2_span": SpanField(e2_start_position, e2_end_position, text_field),
            "input_sentence": MetadataField(sentence),
        }

        if label is not None:
            fields["label"] = LabelField(label)

        if self.use_entity_feature:
            fields["entity_ids"] = ArrayField(np.array([1, 2]))

        return Instance(fields)

    def _read(self, file_path: str):
        for data in self.parser(file_path):
            yield self.text_to_instance(data["sentence"], data["label"])
