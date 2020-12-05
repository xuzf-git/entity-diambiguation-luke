from multilingual.retrieval.reader import BUCCReader

from allennlp.data.tokenizers import SpacyTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer


def test_read():
    reader = BUCCReader(tokenizer=SpacyTokenizer(), token_indexers={"tokens": SingleIdTokenIndexer()})

    instances = reader.read("multilingual/retrieval/tests/fixtures/sample.en")

    assert len(instances) == 2

    fields = instances[0].fields
    assert [t.text for t in fields["tokens"].tokens] == ["This", "is", "a", "test", "."]

    fields = instances[1].fields
    assert [t.text for t in fields["tokens"].tokens] == ["This", "is", "the", "second", "sentence", "."]


def test_stop_words():
    reader = BUCCReader(
        tokenizer=SpacyTokenizer(), token_indexers={"tokens": SingleIdTokenIndexer()}, stop_word_language="english"
    )

    instances = reader.read("multilingual/retrieval/tests/fixtures/sample.en")

    assert len(instances) == 2

    fields = instances[0].fields
    assert [t.text for t in fields["tokens"].tokens] == ["This", "test", "."]

    fields = instances[1].fields
    assert [t.text for t in fields["tokens"].tokens] == ["This", "second", "sentence", "."]
