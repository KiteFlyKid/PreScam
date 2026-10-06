"""AUC, AUPR and Alert Time (AT@FPR) metrics."""

import logging
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

log = logging.getLogger(__name__)


def compute_auc(predictions):
    """Binary ROC-AUC over all (label_binary, score) pairs."""
    valid = [p for p in predictions if p.get("score") is not None]
    if not valid:
        return None
    labels = [p["label_binary"] for p in valid]
    scores = [p["score"] for p in valid]
    if len(set(labels)) < 2:
        log.warning("AUC: only one class present — skipping")
        return None
    return float(roc_auc_score(labels, scores))


def compute_aupr(predictions):
    """Area Under Precision-Recall Curve (average precision)."""
    valid = [p for p in predictions if p.get("score") is not None]
    if not valid:
        return None
    labels = [p["label_binary"] for p in valid]
    scores = [p["score"] for p in valid]
    if len(set(labels)) < 2:
        return None
    return float(average_precision_score(labels, scores))


def _threshold_at_fpr(labels, scores, target_fpr):
    """Return the score threshold that achieves FPR ≤ target_fpr."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    # roc_curve returns increasing FPR; find last threshold where fpr <= target_fpr
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(0, min(idx, len(thresholds) - 1))
    return float(thresholds[idx])


def compute_at(scam_scores, tau=0.85):
    """Alert Time for one conversation.

    Returns T_TM - t_alert (rounds of advance warning) when the score first
    crosses tau, or None if the threshold is never reached.
    """
    sorted_scores = sorted(scam_scores, key=lambda x: x["round_index"])
    if not sorted_scores:
        return None
    N = sorted_scores[0]["total_rounds"]
    for item in sorted_scores:
        if item.get("score") is not None and item["score"] >= tau:
            return N - item["round_index"]
    return None


def compute_at_metrics(predictions, tau=0.85):
    """Aggregate AT metrics across all conversations in the prediction set."""
    by_scam = defaultdict(list)
    for p in predictions:
        if p.get("score") is not None:
            by_scam[p["scam_id"]].append(p)

    at_values, n_fired = [], 0
    n_total = len(by_scam)
    for preds in by_scam.values():
        at = compute_at(preds, tau=tau)
        if at is not None:
            at_values.append(at)
            n_fired += 1

    return {
        "at_mean":        float(np.mean(at_values))   if at_values else None,
        "at_median":      float(np.median(at_values)) if at_values else None,
        "at_firing_rate": n_fired / n_total           if n_total > 0 else None,
        "n_fired":        n_fired,
        "n_total":        n_total,
    }


def compute_at_fpr_metrics(predictions, target_fpr=0.10):
    """AT metrics using a threshold chosen to achieve target FPR on the full prediction set."""
    valid = [p for p in predictions if p.get("score") is not None]
    if not valid:
        return {"at_fpr_tau": None, "at_fpr_mean": None,
                "at_fpr_firing_rate": None, "at_fpr_n_fired": None}
    labels = [p["label_binary"] for p in valid]
    scores = [p["score"] for p in valid]
    if len(set(labels)) < 2:
        return {"at_fpr_tau": None, "at_fpr_mean": None,
                "at_fpr_firing_rate": None, "at_fpr_n_fired": None}

    tau = _threshold_at_fpr(labels, scores, target_fpr)
    at_metrics = compute_at_metrics(predictions, tau=tau)
    return {
        "at_fpr_tau":         tau,
        "at_fpr_mean":        at_metrics["at_mean"],
        "at_fpr_firing_rate": at_metrics["at_firing_rate"],
        "at_fpr_n_fired":     at_metrics["n_fired"],
    }


def evaluate(predictions, at_fpr=0.10):
    """Compute AUC, AUPR and AT@FPR for a set of predictions."""
    return {
        "auc":        compute_auc(predictions),
        "aupr":       compute_aupr(predictions),
        **compute_at_fpr_metrics(predictions, target_fpr=at_fpr),
        "n_examples": len(predictions),
        "n_valid":    sum(1 for p in predictions if p.get("score") is not None),
    }
