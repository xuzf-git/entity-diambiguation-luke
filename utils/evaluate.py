# This code is based on the code obtained from here:
# https://github.com/studio-ousia/luke/blob/master/examples/entity_disambiguation/main.py#L245

import random
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm

def evaluate(args, eval_dataloader, model, entity_vocab, output_file=None):
    predictions = []
    context_entities = []
    labels = []
    documents = []
    mentions = []
    reverse_entity_vocab = {v: k for k, v in entity_vocab.items()}
    for item in tqdm(eval_dataloader, leave=False):  # the batch size must be 1
        inputs = {k: v.to(args.device) for k, v in item.items() if k not in ("document", "mentions", "target_mention_indices")}
        entity_ids = inputs.pop("entity_ids")
        entity_attention_mask = inputs.pop("entity_attention_mask")
        input_entity_ids = entity_ids.new_full(entity_ids.size(), 1)  # [MASK]
        entity_length = entity_ids.size(1)
        with torch.no_grad():
            if not args.no_context_entities:
                result = torch.zeros(entity_length, dtype=torch.long)
                prediction_order = torch.zeros(entity_length, dtype=torch.long)
                for n in range(entity_length):
                    logits = model(entity_ids=input_entity_ids, entity_attention_mask=entity_attention_mask, **inputs)[0]
                    probs = F.softmax(logits, dim=2) * (input_entity_ids == 1).unsqueeze(-1).type_as(logits)
                    max_probs, max_indices = torch.max(probs.squeeze(0), dim=1)
                    if args.context_entity_selection_order == "highest_prob":
                        target_index = torch.argmax(max_probs, dim=0)
                    elif args.context_entity_selection_order == "random":
                        target_index = random.choice((input_entity_ids == 1).squeeze(0).nonzero().view(-1).tolist())
                    elif args.context_entity_selection_order == "natural":
                        target_index = (input_entity_ids == 1).squeeze(0).nonzero().view(-1)[0]
                    input_entity_ids[0, target_index] = max_indices[target_index]
                    result[target_index] = max_indices[target_index]
                    prediction_order[target_index] = n
            else:
                logits = model(entity_ids=input_entity_ids, entity_attention_mask=entity_attention_mask, **inputs)[0]
                result = torch.argmax(logits, dim=2).squeeze(0)

        for index in item["target_mention_indices"][0]:
            predictions.append(result[index].item())
            labels.append(entity_ids[0, index].item())
            documents.append(item["document"][0])
            mentions.append(item["mentions"][0][index])
            if not args.no_context_entities:
                context_entities.append([
                    dict(
                        order=prediction_order[n].item(),
                        prediction=reverse_entity_vocab[result[n].item()],
                        label=mention.title,
                        text=mention.text,
                    ) for n, mention in enumerate(item["mentions"][0]) if prediction_order[n] < prediction_order[index]
                ])
            else:
                context_entities.append([])

    num_correct = 0
    num_mentions = 0
    num_mentions_with_candidates = 0

    eval_predictions = []
    for prediction, label, document, mention, cxt in zip(predictions, labels, documents, mentions, context_entities):
        if prediction == label:
            num_correct += 1

        assert not (mention.candidates and prediction == 0)
        assert label != 0

        num_mentions += 1
        if mention.candidates:
            num_mentions_with_candidates += 1

            eval_predictions.append(
                dict(
                    document_id=document.id,
                    document_words=document.words,
                    document_length=len(document.words),
                    mention_length=len(document.mentions),
                    mention=dict(
                        label=mention.title,
                        text=mention.text,
                        span=(mention.start, mention.end),
                        candidate_length=len(mention.candidates),
                        candidates=[dict(prior_prob=c.prior_prob, title=c.title) for c in mention.candidates],
                    ),
                    prediction=reverse_entity_vocab[prediction],
                    context_entities=cxt,
                ))

    if output_file:
        with open(output_file, "w") as f:
            for obj in eval_predictions:
                f.write(json.dumps(obj) + "\n")

    precision = num_correct / num_mentions_with_candidates
    recall = num_correct / num_mentions
    f1 = 2.0 * precision * recall / (precision + recall)

    print("F1: %.5f" % f1)
    print("Precision: %.5f" % precision)
    print("Recall: %.5f" % recall)

    return dict(precision=precision, recall=recall, f1=f1)
