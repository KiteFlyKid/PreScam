"""Data loading, example building, and text formatting."""

import json
import random
from sklearn.model_selection import train_test_split


def load_data(input_path):
    with open(input_path, encoding="utf-8") as f:
        return json.load(f)


def build_examples(data, min_rounds=3, scammed_only=True,
                   positive_last_k=1, sampling_mode="all", random_seed=42):
    """Build per-turn examples from scam conversations.

    Args:
        positive_last_k: last k rounds of each conversation count as positive
                         (label_binary=1). Default=1 means only t=N-1.
        sampling_mode:
          "all"  — one example per turn t=1..N-1.
                   Positive rate ≈ 1/(N-1); examples within a conversation
                   are nested and non-independent.
          "pair" — one positive + one negative example per conversation,
                   sampled independently. Positive rate = 50%; examples
                   across conversations are independent.
    """
    rng = random.Random(random_seed)
    examples = []

    for entry in data:
        if scammed_only and entry.get("scammed") != 1:
            continue
        rounds = entry.get("engagement", [])
        N = len(rounds)
        if N < min_rounds:
            continue

        # t ∈ {1..N-1}; positive if t >= N - positive_last_k
        pos_pool = list(range(max(1, N - positive_last_k), N))
        neg_pool = list(range(1, max(1, N - positive_last_k)))

        if sampling_mode == "pair":
            if not pos_pool or not neg_pool:
                continue  # can't form a valid pair with this conversation length
            selected = [
                (rng.choice(pos_pool), 1),
                (rng.choice(neg_pool), 0),
            ]
        else:  # "all"
            selected = [
                (t, 1 if t >= N - positive_last_k else 0)
                for t in range(1, N)
            ]

        for t, label in selected:
            examples.append({
                "scam_id":        entry["scam_id"],
                "scam_type":      entry.get("scam_type", ""),
                "initial_contact": entry.get("initial_contact", ""),
                "context_rounds": rounds[:t],
                "round_index":    t,
                "total_rounds":   N,
                "label_binary":   label,
                "label_regression": t / N,
            })

    return examples


def split_data_by_scam_id(data, test_size=0.2, random_seed=42):
    """Split raw data entries 80/20 by scam_id.

    Splitting raw entries (not examples) lets train and test use different
    sampling strategies: e.g. pair mode for training, all mode for testing.
    """
    scam_ids = list({e["scam_id"] for e in data})
    train_ids, test_ids = train_test_split(
        scam_ids, test_size=test_size, random_state=random_seed
    )
    train_set = set(train_ids)
    test_set  = set(test_ids)
    return (
        [e for e in data if e["scam_id"] in train_set],
        [e for e in data if e["scam_id"] in test_set],
    )


def format_context(example, include_victim=True, include_initial_contact=True):
    """Flatten conversation history to a single string for model input."""
    parts = [f"Scam type: {example['scam_type']}"]

    if include_initial_contact and example.get("initial_contact"):
        parts.append(f"Initial contact: {example['initial_contact']}")

    for i, r in enumerate(example["context_rounds"]):
        step_parts = []
        if sa := r.get("scammer_action"):
            step_parts.append(f"Scammer: {sa}")
        if include_victim and (va := r.get("victim_action")):
            step_parts.append(f"Victim: {va}")
        if step_parts:
            parts.append(f"Round {i + 1}: {' | '.join(step_parts)}")

    return "\n".join(parts)
