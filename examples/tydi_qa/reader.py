from typing import List, Tuple, Dict
import json
import bisect
import random
from collections import namedtuple
import numpy as np

from allennlp.data import Instance, Token
from allennlp.data import DatasetReader
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.fields import TextField, LabelField, SpanField, MetadataField, ArrayField

from transformers import AutoTokenizer


from .utils.preproc import read_tydi_examples
from .utils.data_utils import AnswerType
from .utils.byte_utils import byte_to_char_offset, char_to_byte_offset

from examples.utils.wiki_mention_detector import WikiMentionDetector

LANGUAGE_TO_CODE = {
    "ARABIC": "ar",
    "BENGALI": "bn",
    "ENGLISH": "en",
    "FINNISH": "fi",
    "INDONESIAN": "id",
    "JAPANESE": "ja",
    "KOREAN": "ko",
    "RUSSIAN": "ru",
    "TELUGU": "te",
    "SWAHILI": "sw",
    "THAI": "th",
}


def read_passage_answer_candidates(file_path: str) -> Dict[int, List[Dict[str, int]]]:
    result = {}
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            result[data["example_id"]] = data["passage_answer_candidates"]
    return result


@DatasetReader.register("tydi")
class TyDiQAReader(DatasetReader):
    def __init__(
        self,
        transformers_model_name: str,
        max_passages: int = 45,
        max_position: int = 45,
        max_sequence_length: int = 512,
        max_question_length: int = 64,
        document_stride: int = 128,
        include_unknowns_probability: float = -1.0,
        is_evaluation: bool = False,
        mention_detectors: Dict[str, WikiMentionDetector] = None,
        max_num_entity_features: int = 30,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_passages = max_passages
        self.max_position = max_position

        additional_special_tokens = [f"[ContextId={i}]" for i in range(self.max_passages)]
        self.transformers_tokenizer = AutoTokenizer.from_pretrained(
            transformers_model_name, additional_special_tokens=additional_special_tokens
        )
        self.token_indexers = {
            "tokens": PretrainedTransformerIndexer(
                transformers_model_name, tokenizer_kwargs={"additional_special_tokens": additional_special_tokens}
            )
        }

        self.max_sequence_length = max_sequence_length
        self.max_question_length = max_question_length
        self.document_stride = document_stride
        self.include_unknowns_probability = include_unknowns_probability
        self.is_evaluation = is_evaluation

        self.mention_detectors = mention_detectors
        if self.mention_detectors is not None:
            for detector in self.mention_detectors.values():
                detector.set_tokenizer(self.transformers_tokenizer)
        self.max_num_entity_features = max_num_entity_features

    def generate_instances_from_texts(
        self,
        question: str,
        contexts: str,
        answer_type: AnswerType = None,
        start_byte_answer_offset: int = None,
        end_byte_answer_offset: int = None,
        title: str = None,
        language_code: str = None,
    ):

        question_tokens = self.transformers_tokenizer.tokenize(question)
        if len(question_tokens) > self.max_question_length:
            question_tokens = question_tokens[-self.max_question_length :]

        contexts_tokenize_result = self.transformers_tokenizer(
            contexts, return_offsets_mapping=True, add_special_tokens=False
        )
        contexts_tokens = self.transformers_tokenizer.convert_ids_to_tokens(contexts_tokenize_result["input_ids"])
        character_offset_mapping = contexts_tokenize_result["offset_mapping"]
        character_offset_mapping = sanitize_character_offset_mapping(character_offset_mapping)

        if answer_type is not None:
            # convert bytes offsets to token indices
            start_char_answer_offset = byte_to_char_offset(contexts, start_byte_answer_offset)
            end_char_answer_offset = byte_to_char_offset(contexts, end_byte_answer_offset)

            start_token_answer_index = end_token_answer_index = 0
            if answer_type and answer_type != AnswerType.UNKNOWN:
                token_character_start_indices, token_character_end_indices = list(zip(*character_offset_mapping))
                # the token boundaries do not necessarily match the characters of the answer
                # In that case, we choose token spans so that the tokens include all the characters
                start_token_answer_index = (
                    bisect.bisect_right(token_character_start_indices, start_char_answer_offset) - 1
                )
                end_token_answer_index = bisect.bisect_left(token_character_end_indices, end_char_answer_offset)
                assert start_token_answer_index <= end_token_answer_index

        # sliding window
        # DOCUMENT PROCESSING
        # The -3 accounts for
        # 1. [CLS] -- Special BERT class token, which is always first.
        # 2. [SEP] -- Special separator token, placed after question.
        # 3. [SEP] -- Special separator token, placed after article content.
        max_tokens_for_doc = self.max_sequence_length - len(question_tokens) - 3

        # if we use entity feautures, we need room for them.
        if self.mention_detectors is not None:
            max_tokens_for_doc = max_tokens_for_doc - self.max_num_entity_features

        DocSpan = namedtuple("DocSpan", ["start", "length"])
        document_spans = []
        document_span_start_token_offset = 0
        while document_span_start_token_offset < len(contexts_tokens):
            length = min(len(contexts_tokens) - document_span_start_token_offset, max_tokens_for_doc)
            document_spans.append(DocSpan(start=document_span_start_token_offset, length=length))
            if document_span_start_token_offset + length == len(contexts_tokens):
                break
            document_span_start_token_offset += min(length, self.document_stride)

        for document_span_index, document_span in enumerate(document_spans):
            doc_start = document_span.start
            doc_end = document_span.start + document_span.length - 1

            input_tokens = (
                [self.transformers_tokenizer.cls_token]
                + question_tokens
                + [self.transformers_tokenizer.sep_token]
                + contexts_tokens[doc_start : doc_end + 1]
                + [self.transformers_tokenizer.sep_token]
            )
            input_text_field = TextField([Token(t) for t in input_tokens], token_indexers=self.token_indexers)
            assert len(input_text_field) <= self.max_sequence_length

            instance = Instance({"word_ids": input_text_field})

            question_tokens_offset = len(question_tokens) + 2
            doc_offset = doc_start - question_tokens_offset  # one for CLS, one for SEP.
            if answer_type is not None:
                contains_an_annotation = doc_start <= start_token_answer_index and end_token_answer_index <= doc_end
                if (not contains_an_annotation) or answer_type == AnswerType.UNKNOWN:
                    # If an example has unknown answer type or does not contain the answer
                    # span, then we only include it with probability --include_unknowns.
                    # When we include an example with unknown answer type, we set the first
                    # token of the passage to be the annotated short span.
                    if self.include_unknowns_probability < 0 or random.random() > self.include_unknowns_probability:
                        continue

                    # when no answer found in the current document span, the start/end index point to the [CLS] token
                    input_start_answer_index = input_end_answer_index = 0
                    answer_type = AnswerType.UNKNOWN
                else:
                    input_start_answer_index = start_token_answer_index - doc_offset
                    input_end_answer_index = end_token_answer_index - doc_offset

                instance.add_field("answer_type", LabelField(answer_type.name, label_namespace="answer_type"))
                instance.add_field(
                    "answer_span", SpanField(input_start_answer_index, input_end_answer_index, input_text_field)
                )

            if self.mention_detectors is not None:
                assert title is not None and language_code is not None
                for field_name, field in self.get_entity_features(input_tokens, title=title, language=language_code):
                    instance.add_field(field_name, field)

            if self.is_evaluation:
                token_to_contexts_byte_mapping = []
                prev_start = -1
                for i, t in enumerate(input_tokens):
                    if i <= question_tokens_offset or len(character_offset_mapping) <= i + doc_offset:
                        token_byte_span = (-1, -1)
                    else:
                        char_start, char_end = character_offset_mapping[i + doc_offset]
                        byte_start = char_to_byte_offset(contexts, char_start)
                        byte_end = char_to_byte_offset(contexts, char_end)
                        # deal with a weired property of character_offset_mapping
                        # it produces incorrect offsets for trailing whitespaces/new line characters
                        if byte_start == prev_start:
                            token_to_contexts_byte_mapping[-1] = (-1, -1)
                        token_byte_span = (byte_start, byte_end)
                        prev_start = byte_start

                    token_to_contexts_byte_mapping.append(token_byte_span)

                instance.add_field(
                    "metadata",
                    MetadataField(
                        {
                            "contexts": contexts,
                            "token_to_contexts_byte_mapping": token_to_contexts_byte_mapping,
                            "document_span_index": document_span_index,
                        }
                    ),
                )

            yield instance

    def _read(self, file_path: str):
        tydi_examples = read_tydi_examples(
            file_path,
            is_training=True,
            max_position=self.max_position,
            max_passages=self.max_passages,
            fail_on_invalid=True,
        )

        if self.is_evaluation:
            passage_answer_candidates = read_passage_answer_candidates(file_path)

        for example in tydi_examples:
            metadata = MetadataField(
                {
                    "example_id": str(example.example_id),
                    "language": example.language_id.name,
                    "document_title": example.document_title,
                }
            )
            for instance in self.generate_instances_from_texts(
                example.question,
                example.contexts,
                example.answer.type if example.answer else AnswerType.UNKNOWN,
                example.start_byte_offset,
                example.end_byte_offset,
            ):
                if self.is_evaluation:
                    instance.add_field("example_metadata", metadata)
                    instance["metadata"].metadata.update(
                        {
                            "example_id": example.example_id,
                            "context_to_plaintext_offset": example.context_to_plaintext_offset,
                            "plaintext": example.plaintext,
                            "answer_text": example.answer.text if example.answer else None,
                            "passage_answer_candidates": passage_answer_candidates[example.example_id],
                            "language": example.language_id.name.lower(),
                        }
                    )

                yield instance

    def get_entity_features(self, tokens: List[Token], title: str, language: str):
        assert language is not None
        mention_detector = self.mention_detectors[language]
        mentions = mention_detector.detect_mentions(tokens, title, language)[: self.max_num_entity_features]
        entity_features = mention_detector.mentions_to_entity_features(tokens, mentions)

        entity_feature_fields = {}
        for name, feature in entity_features.items():
            entity_feature_fields[name] = ArrayField(np.array(feature), padding_value=0)
        return entity_feature_fields


def sanitize_character_offset_mapping(character_offset_mapping: List[Tuple[int, int]]):
    """
    Sentence Piece does not produce correct mapping for tokenized whitespaces ("▁").
    We replace the incorrect span with (previous span end, next span start).
    """
    for i in range(1, len(character_offset_mapping) - 1):
        if character_offset_mapping[i][0] == character_offset_mapping[i + 1][0]:
            character_offset_mapping[i] = (character_offset_mapping[i - 1][1], character_offset_mapping[i + 1][0])
    return character_offset_mapping
