from examples.bucc.reader import BUCCReader

from allennlp.data.tokenizers import SpacyTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer

TEST_FILE = "multilingual/tasks/bucc/tests/fixtures/sample.en"


def test_read():
    reader = BUCCReader(tokenizer=SpacyTokenizer(), token_indexers={"tokens": SingleIdTokenIndexer()})

    instances = reader.read(TEST_FILE)

    assert len(instances) == 2

    fields = instances[0].fields
    assert [t.text for t in fields["tokens"].tokens] == ["This", "is", "a", "test", "."]
    assert fields["index"].metadata == "en-000000001"

    fields = instances[1].fields
    assert [t.text for t in fields["tokens"].tokens] == ["This", "is", "the", "second", "sentence", "."]
    assert fields["index"].metadata == "en-000000002"


def test_stop_words():
    reader = BUCCReader(
        tokenizer=SpacyTokenizer(), token_indexers={"tokens": SingleIdTokenIndexer()}, stop_word_language="english"
    )

    instances = reader.read(TEST_FILE)

    assert len(instances) == 2

    fields = instances[0].fields
    assert [t.text for t in fields["tokens"].tokens] == ["This", "test", "."]
    assert fields["index"].metadata == "en-000000001"

    fields = instances[1].fields
    assert [t.text for t in fields["tokens"].tokens] == ["This", "second", "sentence", "."]
    assert fields["index"].metadata == "en-000000002"
