from typing import Dict, List, Tuple
import json


from allennlp.data import DatasetReader, Instance, Tokenizer, TokenIndexer
from allennlp.data.fields import MetadataField, TextField


def parse_lareqa_file(file_path: str):
    data = json.load(open(file_path, "r"))
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            yield from parse_paragraph(paragraph)


def parse_paragraph(paragraph: Dict):
    def _get_sentence_index(sentence_spans: List[Tuple[int, int]], start_index: int) -> int:
        last_span_end = sentence_spans[-1][1]
        assert start_index < last_span_end

        for index, (span_start, span_end) in enumerate(sentence_spans):
            if span_start <= start_index <= span_end:
                return index

    for q in paragraph["qas"]:
        for answer in q["answers"]:
            sentence_index = _get_sentence_index(paragraph["sentence_breaks"], answer["answer_start"])
            yield {"question": q["question"], "answer": paragraph["sentences"][sentence_index], "idx": q["id"]}


@DatasetReader.register("lareqa")
class LAReQAReader(DatasetReader):
    def __init__(self, tokenizer: Tokenizer, token_indexers: Dict[str, TokenIndexer], **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers

    def text_to_instance(self, question: str, answer: str, idx: str) -> Instance:
        question_tokens = self.tokenizer.tokenize(question)
        answer_tokens = self.tokenizer.tokenize(answer)
        fields = {
            "question": TextField(question_tokens, self.token_indexers),
            "answer": TextField(answer_tokens, self.token_indexers),
            "index": MetadataField(idx),
        }
        return Instance(fields)

    def _read(self, file_path: str):
        for question_answer_pair in parse_lareqa_file(file_path):
            yield self.text_to_instance(**question_answer_pair)
