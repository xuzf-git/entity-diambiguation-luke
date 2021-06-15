from allennlp.data.fields import MetadataField

from allennlp.data import DatasetReader, Instance
from allennlp_models.rc.dataset_readers import TransformerSquadReader


@DatasetReader.register("transformers_squad")
class SquadReader(TransformerSquadReader):
    def _read(self, file_path: str):

        for instance in super()._read(file_path):
            input_tokens = [t.text for t in instance["question_with_context"]]
            yield Instance(
                {
                    "question_with_context": instance["question_with_context"],
                    "answer_span": instance["answer_span"],
                    "metadata": MetadataField(
                        {"input_tokens": input_tokens, "example_id": instance["metadata"].metadata["id"]}
                    ),
                }
            )
