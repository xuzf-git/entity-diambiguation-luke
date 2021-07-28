# coding=utf-8
# Copyright 2020 The Google Research Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Performs model-specific preprocessing.

This includes tokenization and adding special tokens to the input.

This module does not have any dependencies on TensorFlow and should be re-usable
within your favorite ML/DL framework.
"""
from typing import Dict
import collections
import functools
import glob
import json

from absl import logging
import examples_allennlp.reading_comprehension.utils.data_utils as data_utils


def create_entry_from_json(
    json_dict: Dict, max_passages: int = 45, max_position: int = 45, fail_on_invalid: bool = True
):
    """Creates an TyDi 'entry' from the raw JSON.

  The 'TyDiEntry' dict is an intermediate format that is later converted into
  the main `TyDiExample` format.

  This function looks up the chunks of text that are candidates for the passage
  answer task, inserts special context tokens such as "[ContextId=0]", and
  creates a byte index to byte index mapping between the document plaintext
  and the concatenation of the passage candidates (these could potentially
  exclude parts of the plaintext document and also include the special tokens).

  In the returned entry, `contexts` includes only the candidate passages and
  has special tokens such as [ContextId=0] added. `span_start` and `span_end`
  are byte-wise indices into `contexts` (not the original corpus plaintext).

  Args:
    json_dict: A single JSONL line, deserialized into a dict.
    max_passages: see FLAGS.max_passages.
    max_position: see FLAGS.max_position.
    fail_on_invalid: Immediately stop if an error is found?

  Returns:
    If a failure was encountered and `fail_on_invalid=False`, then returns
    an empty `dict`. Otherwise returns:
    'TyDiEntry' type: a dict-based format consumed by downstream functions:
    entry = {
        "name": str,
        "id": str,
        "language": str,
        "question": {"input_text": str},
        "answer": {
          "candidate_id": annotated_idx,
          "span_text": "",
          "span_start": -1,
          "span_end": -1,
          "input_text": "passage",
        }
        "has_correct_context": bool,
        # Includes special tokens appended.
        "contexts": str,
        # Context index to byte offset in `contexts`.
        "context_to_plaintext_offset": Dict[int, int],
        "plaintext" = json_dict["document_plaintext"]
    }
  """

    add_candidate_types_and_positions(json_dict, max_position)
    for passage_answer in json_dict["passage_answer_candidates"]:
        if passage_answer["plaintext_start_byte"] == -1 or passage_answer["plaintext_end_byte"] == -1:
            return {}

    # annotated_idx: index of the first annotated context, -1 if null.
    # annotated_min_ans: minimal answer start and end char offsets,
    #                    (-1, -1) if null.
    annotation, annotated_idx, annotated_min_ans = data_utils.get_first_annotation(json_dict, max_passages)
    question = {"input_text": json_dict["question_text"]}
    answer = {
        "candidate_id": annotated_idx,
        "span_text": "",
        "span_start": -1,
        "span_end": -1,
        "input_text": "passage",
    }

    # Yes/no answers are added in the input text.
    if annotation is not None:
        assert annotation["yes_no_answer"] in ("YES", "NO", "NONE")
        if annotation["yes_no_answer"] in ("YES", "NO"):
            answer["input_text"] = annotation["yes_no_answer"].lower()

    # Add a minimal answer if one was found.
    if annotated_min_ans != (-1, -1):
        answer["input_text"] = "minimal"
        span_text = data_utils.get_candidate_text(json_dict, annotated_idx).text

        try:
            answer["span_text"] = data_utils.byte_slice(span_text, annotated_min_ans[0], annotated_min_ans[1])
        except UnicodeDecodeError:
            logging.error("UnicodeDecodeError for example: %s", json_dict["example_id"])
            if fail_on_invalid:
                raise
            return {}
        # local (passage) byte offset
        answer["span_start"] = annotated_min_ans[0]
        answer["span_end"] = annotated_min_ans[1]
        try:
            expected_answer_text = data_utils.get_text_span(
                json_dict,
                {
                    "plaintext_start_byte": annotation["minimal_answer"]["plaintext_start_byte"],
                    "plaintext_end_byte": annotation["minimal_answer"]["plaintext_end_byte"],
                },
            ).text
        except UnicodeDecodeError:
            logging.error("UnicodeDecodeError for example: %s", json_dict["example_id"])
            if fail_on_invalid:
                raise
            return {}
        if expected_answer_text != answer["span_text"]:
            error_message = "Extracted answer did not match expected answer:" "'{}' vs '{}'".format(
                expected_answer_text, answer["span_text"]
            )
            if fail_on_invalid:
                raise ValueError(error_message)
            else:
                logging.warn(error_message)
                return {}

    # Add a passage answer if one was found
    elif annotation and annotation["passage_answer"]["candidate_index"] >= 0:
        answer["input_text"] = "passage"
        answer["span_text"] = data_utils.get_candidate_text(json_dict, annotated_idx).text
        answer["span_start"] = 0
        answer["span_end"] = data_utils.byte_len(answer["span_text"])

    context_idxs = []
    context_list = []
    for idx, _ in data_utils.candidates_iter(json_dict):
        context = {"id": idx, "type": get_candidate_type_and_position(json_dict, idx)}
        # Get list of all byte positions of the candidate and its plaintext.
        # Unpack `TextSpan` tuple.
        context["text_map"], context["text"] = data_utils.get_candidate_text(json_dict, idx)
        if not context["text"]:
            logging.error("ERROR: Found example with empty context %d.", idx)
            if fail_on_invalid:
                raise ValueError("ERROR: Found example with empty context {}.".format(idx))
            return {}
        context_idxs.append(idx)
        context_list.append(context)
        if len(context_list) >= max_passages:
            break

    # Assemble the entry to be returned.
    entry = {
        "name": json_dict["document_title"],
        "id": str(json_dict["example_id"]),
        "language": json_dict["language"],
        "question": question,
        "answer": answer,
        "has_correct_context": annotated_idx in context_idxs,
        "document_title": json_dict["document_title"],
    }
    all_contexts_with_tokens = []
    # `offset` is a byte offset relative to `contexts` (concatenated candidate
    # passages with special tokens added).
    offset = 0
    context_to_plaintext_offset = []
    for idx, context in zip(context_idxs, context_list):
        special_token = "[ContextId={}]".format(context["id"])
        all_contexts_with_tokens.append(special_token)
        context_to_plaintext_offset.append([-1] * data_utils.byte_len(special_token))
        # Account for the special token and its trailing space (due to the join
        # operation below)
        offset += data_utils.byte_len(special_token) + 1

        if context["id"] == annotated_idx:
            answer["span_start"] += offset
            answer["span_end"] += offset
        if context["text"]:
            all_contexts_with_tokens.append(context["text"])
            # Account for the text and its trailing space (due to the join
            # operation below)
            offset += data_utils.byte_len(context["text"]) + 1
            context_to_plaintext_offset.append(context["text_map"])
        else:
            if fail_on_invalid:
                raise ValueError("Found example with empty context.")

    # When we join the contexts together with spaces below, we'll add an extra
    # byte to each one, so we have to account for these by adding a -1 (no
    # assigned wordpiece) index at each *boundary*. It's easier to do this here
    # than above since we don't want to accidentally add extra indices after the
    # last context.
    context_to_plaintext_offset = functools.reduce(lambda a, b: a + [-1] + b, context_to_plaintext_offset)

    entry["contexts"] = " ".join(all_contexts_with_tokens)
    entry["context_to_plaintext_offset"] = context_to_plaintext_offset
    entry["plaintext"] = json_dict["document_plaintext"]

    if annotated_idx in context_idxs:
        try:
            expected = data_utils.byte_slice(entry["contexts"], answer["span_start"], answer["span_end"])
        except UnicodeDecodeError:
            logging.error("UnicodeDecodeError for example: %s", json_dict["example_id"])
            if fail_on_invalid:
                raise
            return {}
        # This is a sanity check to ensure that the calculated start and end
        # indices match the reported span text. If this assert fails, it is likely
        # a bug in the data preparation code above. (expected, answer["span_text"])
        if expected != answer["span_text"]:
            logging.warn("*** pruned example id: %d ***", json_dict["example_id"])
            logging.warn("*** %s, %s ***", expected, answer["span_text"])
            return {}
    return entry


def add_candidate_types_and_positions(json_dict: Dict, max_position: int):
    """Adds type and position info to each candidate in the document."""
    count = 0
    for _, cand in data_utils.candidates_iter(json_dict):
        if count < max_position:
            count += 1
        cand["type_and_position"] = "[Paragraph=%d]" % count


def get_candidate_type_and_position(json_dict, idx):
    """Returns type and position info for the candidate at the given index."""
    if idx == -1:
        return "[NoLongAnswer]"
    else:
        # Gets the 'type_and_position' for this candidate as added by
        # `add_candidate_types_and_positions`. Note that this key is not present
        # in the original TyDi QA corpus.
        return json_dict["passage_answer_candidates"][idx]["type_and_position"]


def find_nearest_wordpiece_index(offset_index, offset_to_wp, scan_right=True):
    """According to offset_to_wp dictionary, find the wordpiece index for offset.

  Some offsets do not have mapping to word piece index if they are delimited.
  If scan_right is True, we return the word piece index of nearest right byte,
  nearest left byte otherwise.

  Args:
    offset_index: the target byte offset.
    offset_to_wp: a dictionary mapping from byte offset to wordpiece index.
    scan_right: When there is no valid wordpiece for the offset_index, will
      consider offset_index+i if this is set to True, offset_index-i otherwise.

  Returns:
    The index of the nearest word piece of `offset_index`
    or -1 if no match is possible.
  """

    for i in range(0, len(offset_to_wp.items())):
        next_ind = offset_index + i if scan_right else offset_index - i
        if next_ind >= 0 and next_ind in offset_to_wp:
            return_ind = offset_to_wp[next_ind]
            # offset has a match.
            if return_ind > -1:
                return return_ind
    return -1


def create_mapping(start_offsets, end_offsets, context_to_plaintext_offset):
    """Creates a mapping from context offsets to plaintext offsets.

  Args:
    start_offsets: List of offsets relative to a TyDi entry's `contexts`.
    end_offsets: List of offsets relative to a TyDi entry's `contexts`.
    context_to_plaintext_offset: Dict mapping `contexts` offsets to plaintext
      offsets.

  Returns:
    List of offsets relative to the original corpus plaintext.
  """

    plaintext_start_offsets = [context_to_plaintext_offset[i] if i >= 0 else -1 for i in start_offsets]
    plaintext_end_offsets = [context_to_plaintext_offset[i] if i >= 0 else -1 for i in end_offsets]
    return plaintext_start_offsets, plaintext_end_offsets


def read_tydi_examples(
    input_file: str,
    is_training: bool = True,
    max_passages: int = 45,
    max_position: int = 45,
    fail_on_invalid: bool = True,
):
    """Read a TyDi json file into a list of `TyDiExample`.

  Delegates to `preproc.create_entry_from_json` to add special tokens to
  input and handle character offset tracking.

  Args:
    input_file: Path or glob to input JSONL files to be read (possibly gzipped).
    is_training: Should we create training samples? (as opposed to eval
      samples).
    max_passages: See FLAGS.max_passages.
    max_position: See FLAGS.max_position.
    fail_on_invalid: Should we immediately stop processing if an error is
      encountered?
    open_fn: A function that returns a file object given a path. Usually
      `tf_io.gopen`; could be standard Python `open` if using this module
      outside Tensorflow.

  Yields:
    `TyDiExample`s
  """
    input_paths = glob.glob(input_file)
    if not input_paths:
        raise ValueError("No paths matching glob '{}'".format(input_file))

    non_valid_count = 0
    n = 0
    for path in input_paths:
        logging.info("Reading: %s", path)
        with open(path) as input_file:
            logging.info(path)
            for line in input_file:
                json_dict = json.loads(line, object_pairs_hook=collections.OrderedDict)
                entry = create_entry_from_json(
                    json_dict, max_passages=max_passages, max_position=max_position, fail_on_invalid=fail_on_invalid,
                )
                if entry:
                    tydi_example = data_utils.to_tydi_example(entry, is_training)
                    n += 1
                    yield tydi_example
                else:
                    if fail_on_invalid:
                        raise ValueError("Found invalid example.")
                    non_valid_count += 1

    if n == 0:
        raise ValueError("No surviving examples from input_file '{}'".format(input_file))

    logging.info("*** # surviving examples %d ***", n)
    logging.info("*** # pruned examples %d ***", non_valid_count)
