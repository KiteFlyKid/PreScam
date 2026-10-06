"""
Text-level metrics: BERTScore and ROUGE-L.

For both metrics, we concatenate all generated scammer_actions into one string
and all ground-truth scammer_actions into another, then compute the score
on the concatenated pair.
"""

from bert_score import score as bert_score
from rouge_score import rouge_scorer


def _concat_actions(turns_or_text, mode="structural"):
    """Concatenate scammer actions from structural turns or plain text into a single string."""
    if mode == "plain" or isinstance(turns_or_text, str):
        return turns_or_text
    if not turns_or_text:
        return ""
    actions = []
    for t in turns_or_text:
        if isinstance(t, dict):
            actions.append(t.get("scammer_action", ""))
        else:
            actions.append(str(t))
    return " ".join(actions)


def compute_text_metrics(pred):
    """Compute BERTScore F1 and ROUGE-L for a single prediction.

    Args:
        pred: dict with "generated", "ground_truth", and optionally "output_mode"

    Returns:
        {"bert_score_f1": float, "rouge_l": float} or None if missing data
    """
    if not pred.get("generated") or not pred.get("ground_truth"):
        return None

    mode = pred.get("output_mode", "structural")
    gen_text = _concat_actions(pred["generated"], mode)
    gt_text = _concat_actions(pred["ground_truth"], mode="structural")  # GT is always structural

    if not gen_text.strip() or not gt_text.strip():
        return None

    # BERTScore (roberta-large is the recommended default, 512 token limit)
    _, _, f1 = bert_score(
        [gen_text], [gt_text],
        lang="en",
        model_type="roberta-large",
        verbose=False,
    )
    bert_f1 = f1.item()

    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_result = scorer.score(gt_text, gen_text)
    rouge_l = rouge_result["rougeL"].fmeasure

    return {
        "bert_score_f1": bert_f1,
        "rouge_l": rouge_l,
    }
