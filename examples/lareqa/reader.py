from typing import Dict, List, Tuple
import json
import itertools
import glob

from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data import DatasetReader, Instance, Tokenizer, TokenIndexer
from allennlp.data.fields import MetadataField, TextField

from luke.utils.sentence_tokenizer import SentenceTokenizer, ICUSentenceTokenizer

from .utils import WikiMentionDetector


class LAReQAParser:
    def __init__(self, mode: str = "lareqa", sentence_splitter: SentenceTokenizer = None):
        assert mode in {"lareqa", "squad"}

        self.mode = mode
        self.sentence_splitter = sentence_splitter or ICUSentenceTokenizer()

    def __call__(self, file_path: str):
        data = json.load(open(file_path, "r"))
        for article in data["data"]:
            for paragraph in article["paragraphs"]:
                if self.mode == "lareqa":
                    yield from self.parse_lareqa_paragraph(paragraph, title=article["title"])
                elif self.mode == "squad":
                    yield from self.parse_squad_paragraph(paragraph, title=article["title"])
                else:
                    raise ValueError(f"self.mode = {self.mode}")

    @staticmethod
    def get_sentence_index(sentence_spans: List[Tuple[int, int]], start_index: int) -> int:
        last_span_end = sentence_spans[-1][1]
        assert start_index < last_span_end

        for index, (span_start, span_end) in enumerate(sentence_spans):
            if span_start <= start_index <= span_end:
                return index

    def parse_lareqa_paragraph(self, paragraph: Dict, title: str = None):

        for q in paragraph["qas"]:
            for answer in q["answers"]:
                sentence_index = self.get_sentence_index(paragraph["sentence_breaks"], answer["answer_start"])
                yield {
                    "question": q["question"],
                    "answer": paragraph["sentences"][sentence_index],
                    "context_paragraph": paragraph["sentences"],
                    "idx": q["id"],
                    "title": title,
                }

    def parse_squad_paragraph(self, paragraph: Dict, title: str = None):
        sentence_breaks = self.sentence_splitter.span_tokenize(paragraph["context"])
        sentences = [paragraph["context"][s:e] for s, e in sentence_breaks]
        for q in paragraph["qas"]:
            for answer in q["answers"]:
                sentence_index = self.get_sentence_index(sentence_breaks, answer["answer_start"])
                yield {
                    "question": q["question"],
                    "answer": sentences[sentence_index],
                    "context_paragraph": sentences,
                    "idx": q["id"],
                    "title": title,
                }


@DatasetReader.register("lareqa")
class LAReQAReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer,
        token_indexers: Dict[str, TokenIndexer],
        mode: str = "lareqa",
        max_sequence_length: int = 512,
        wiki_mention_detector: WikiMentionDetector = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.parser = LAReQAParser(mode=mode)

        self.max_sequence_length = max_sequence_length

        self.wiki_mention_detector = wiki_mention_detector
        if self.wiki_mention_detector is not None:
            self.wiki_mention_detector.set_tokenizer(tokenizer)

    def text_to_instance(self, question: str, answer: str, context_paragraph: List[str], idx: str) -> Instance:
        question_tokens = self.tokenizer.tokenize(question)
        answer_tokens = self.tokenizer.tokenize(answer)

        context_tokens = [self.tokenizer.tokenize(s) for s in context_paragraph]
        if isinstance(self.tokenizer, PretrainedTransformerTokenizer):
            # drop the [CLS] and [SEP] token
            context_tokens = [tokens[1:-1] for tokens in context_tokens]
            # append [SEP] to last
            context_tokens[-1].append(answer_tokens[-1])

        context_tokens = list(itertools.chain(*context_tokens))
        for token in context_tokens:
            token.type_id = 1

        fields = {
            "question": TextField(question_tokens, self.token_indexers),
            "answer": TextField(answer_tokens + context_tokens, self.token_indexers),
            "ids": MetadataField(idx),
        }

        return Instance(fields)

    def _read(self, file_path: str):
        file_path_list = glob.glob(file_path)

        if len(file_path_list) == 0:
            raise ValueError(f"``{file_path}`` matches no file.")

        for file_path in file_path_list:
            for question_answer_pair in self.parser(file_path):
                instance = self.text_to_instance(**question_answer_pair)

                if (
                    len(instance["answer"]) > self.max_sequence_length
                    or len(instance["question"]) > self.max_sequence_length
                ):
                    continue

                yield instance
