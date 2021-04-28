from typing import Dict, List, Tuple
import json
import itertools
import glob
from pathlib import Path
import numpy as np

from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data import DatasetReader, Instance, Tokenizer, TokenIndexer, Token
from allennlp.data.fields import MetadataField, TextField, ArrayField

from luke.utils.sentence_tokenizer import SentenceTokenizer
from .utils.sentence_breaker import SQuADSentenceTokenizer

from examples.utils.wiki_mention_detector import WikiMentionDetector


class LAReQAParser:
    def __init__(self, mode: str = "lareqa", sentence_splitter: SentenceTokenizer = None):
        assert mode in {"lareqa", "squad"}

        self.mode = mode
        self.sentence_splitter = sentence_splitter or SQuADSentenceTokenizer()

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
        max_query_length: int = 128,
        max_answer_length: int = 512,
        wiki_mention_detector: WikiMentionDetector = None,
        use_segment_type_id: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.parser = LAReQAParser(mode=mode)

        self.max_query_length = max_query_length
        self.max_answer_length = max_answer_length

        self.wiki_mention_detector = wiki_mention_detector
        if self.wiki_mention_detector is not None:
            if not isinstance(tokenizer, PretrainedTransformerTokenizer):
                raise ValueError("WikiMentionDetector is only compatible with PretrainedTransformerTokenizer.")
            self.wiki_mention_detector.set_tokenizer(tokenizer.tokenizer)

        self.use_segment_type_id = use_segment_type_id

    def text_to_instance(
        self, question: str, answer: str, context_paragraph: List[str], idx: str, title: str, language: str = None
    ) -> Instance:
        question_tokens = self.tokenizer.tokenize(question)
        answer_tokens = self.tokenizer.tokenize(answer)

        context_tokens = [self.tokenizer.tokenize(s) for s in context_paragraph]
        if isinstance(self.tokenizer, PretrainedTransformerTokenizer):
            # drop the [CLS] and [SEP] token
            context_tokens = [tokens[1:-1] for tokens in context_tokens]
            # append [SEP] to last
            context_tokens[-1].append(answer_tokens[-1])

        context_tokens = list(itertools.chain(*context_tokens))

        if self.use_segment_type_id:
            for token in context_tokens:
                token.type_id = 1

        answer_context_tokens = answer_tokens + context_tokens

        fields = {}
        max_query_length = self.max_query_length
        max_answer_length = self.max_answer_length
        if self.wiki_mention_detector is not None:
            answer_entity_fields = self.get_entity_features(answer_context_tokens, title, language=language)
            fields.update({"answer_" + k: v for k, v in answer_entity_fields.items()})

            question_entity_fields = self.get_entity_features(question_tokens, title, language=language)
            fields.update({"question_" + k: v for k, v in question_entity_fields.items()})

        fields = {
            "question": TextField(question_tokens[:max_query_length], self.token_indexers),
            "answer": TextField(answer_context_tokens[:max_answer_length], self.token_indexers),
            "ids": MetadataField(idx),
        }

        return Instance(fields)

    def get_entity_features(self, tokens: List[Token], title: str, language: str):
        assert language is not None
        mentions = self.wiki_mention_detector.detect_mentions(tokens, title, language)
        entity_features = self.wiki_mention_detector.mentions_to_entity_features(tokens, mentions)

        entity_feature_fields = {}
        for name, feature in entity_features.items():
            entity_feature_fields[name] = ArrayField(np.array(feature), padding_value=0)
        return entity_feature_fields

    def _read(self, file_path: str):
        file_path_list = glob.glob(file_path)

        if len(file_path_list) == 0:
            raise ValueError(f"``{file_path}`` matches no file.")

        for file_path in file_path_list:

            if self.parser.mode == "lareqa":
                # we asusme the filename is like "en.json"
                language = Path(file_path).stem
            elif self.parser.mode == "squad":
                language = "en"

            for question_answer_pair in self.parser(file_path):
                instance = self.text_to_instance(**question_answer_pair, language=language)
                yield instance
