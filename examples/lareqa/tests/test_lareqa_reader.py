from examples.lareqa.reader import LAReQAReader, parse_lareqa_file

from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer


TEST_FILE = "examples/lareqa/tests/fixtures/test.json"


def test_parse_lareqa_file():

    instances = list(parse_lareqa_file(TEST_FILE))

    assert len(instances) == 2

    instance = instances[0]
    assert instance["question"] == "What is 1 plus 1?"
    assert instance["answer"] == "One plus one is 2."
    assert instance["context_paragraph"] == ["One plus one is 2.", "I like penguins."]
    assert instance["idx"] == "test00"

    instance = instances[1]
    assert instance["question"] == "What is your favorite animal?"
    assert instance["answer"] == "I like penguins."
    assert instance["context_paragraph"] == ["One plus one is 2.", "I like penguins."]
    assert instance["idx"] == "test01"


def test_read():
    reader = LAReQAReader(
        tokenizer=PretrainedTransformerTokenizer(model_name="bert-base-cased"),
        token_indexers={"tokens": SingleIdTokenIndexer()}
    )

    instances = reader.read(TEST_FILE)

    assert len(instances) == 2

    fields = instances[0].fields
    assert [t.text for t in fields["question"].tokens] == ["[CLS]", "What", "is", "1", "plus", "1", "?", "[SEP]"]
    assert [t.text for t in fields["answer"].tokens] == ["[CLS]", "One", "plus", "one", "is", "2", ".", "[SEP]",
                                                         "One", "plus", "one", "is", "2", ".",
                                                         "I", "like", "pen", "##guin", "##s", ".", "[SEP]"]
    assert fields["index"].metadata == "test00"

    fields = instances[1].fields
    assert [t.text for t in fields["question"].tokens] == ["[CLS]", "What", "is", "your", "favorite", "animal", "?", "[SEP]"]
    assert [t.text for t in fields["answer"].tokens] == ["[CLS]", "I", "like", "pen", "##guin", "##s", ".", "[SEP]",
                                                         "One", "plus", "one", "is", "2", ".",
                                                         "I", "like", "pen", "##guin", "##s", ".", "[SEP]"]
    assert fields["index"].metadata == "test01"

