from typing import List
import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.data import TextFieldTensors

from examples.utils.retrieval.models import Seq2VecEncoder
from examples.utils.retrieval.embedding_similarity_loss import EmbeddingSimilarityLoss

from examples.lareqa.metrics.mean_average_precision import MeanAveragePrecision


@Model.register("dual_encoder_retrieval")
class DualEncoder(Model):
    """
    Implement the dual-encoder architecture described in
    ``LAReQA: Language-Agnostic Answer Retrieval from a Multilingual Pool``
    (https://www.aclweb.org/anthology/2020.emnlp-main.477)
    """

    def __init__(
        self,
        vocab: Vocabulary,
        encoder: Seq2VecEncoder,
        criterion: EmbeddingSimilarityLoss,
        evaluate_top_k: int = 1,
        use_similarity_scale: bool = True,
        normalize_embeddings: bool = True,
    ):
        super().__init__(vocab=vocab)

        self.encoder = encoder
        self.criterion = criterion

        self.metrics = {"mAP": MeanAveragePrecision(k=evaluate_top_k)}

        self.similarity_scale = torch.nn.Parameter(torch.ones(1)) if use_similarity_scale else None
        self.normalize_embeddings = normalize_embeddings

    def forward(
        self,
        question: TextFieldTensors,
        answer: TextFieldTensors,
        ids: List[str],
        question_entity_ids: torch.LongTensor = None,
        question_entity_position_ids: torch.LongTensor = None,
        question_entity_segment_ids: torch.LongTensor = None,
        question_entity_attention_mask: torch.LongTensor = None,
        answer_entity_ids: torch.LongTensor = None,
        answer_entity_position_ids: torch.LongTensor = None,
        answer_entity_segment_ids: torch.LongTensor = None,
        answer_entity_attention_mask: torch.LongTensor = None,
        **kwargs
    ):

        if question_entity_ids is not None:
            question["tokens"]["entity_ids"] = question_entity_ids
            question["tokens"]["entity_position_ids"] = question_entity_position_ids
            question["tokens"]["entity_segment_ids"] = question_entity_segment_ids
            question["tokens"]["entity_attention_mask"] = question_entity_attention_mask

        if answer_entity_ids is not None:
            answer["tokens"]["entity_ids"] = answer_entity_ids
            answer["tokens"]["entity_position_ids"] = answer_entity_position_ids
            answer["tokens"]["entity_segment_ids"] = answer_entity_segment_ids
            answer["tokens"]["entity_attention_mask"] = answer_entity_attention_mask

        question_embeddings = self.encoder(question)
        answer_embeddings = self.encoder(answer)

        if self.normalize_embeddings:
            question_embeddings = question_embeddings / torch.norm(question_embeddings, dim=1, keepdim=True)
            answer_embeddings = answer_embeddings / torch.norm(answer_embeddings, dim=1, keepdim=True)

        similarity_matrix = torch.mm(question_embeddings, answer_embeddings.transpose(0, 1))

        if self.similarity_scale is not None:
            similarity_matrix = similarity_matrix * self.similarity_scale

        loss = self.criterion.forward(similarity_matrix)

        if not self.training:
            self.metrics["mAP"](question_embeddings, answer_embeddings, query_ids=ids, target_ids=ids)

        return {"loss": loss}

    def get_metrics(self, reset: bool = False):
        return {k: metric.get_metric(reset) for k, metric in self.metrics.items()}
