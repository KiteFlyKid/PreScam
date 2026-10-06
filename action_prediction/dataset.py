import json
from torch.utils.data import Dataset


def load_llm_labels(labels_path):
    """Load the LLM-labelled boundary turn (detected_turn, 1-based) per scam_id."""
    labels = {}
    with open(labels_path) as f:
        for line in f:
            row = json.loads(line)
            labels[row["scam_id"]] = row["detected_turn"]
    return labels


def build_text(record, turn_idx, cfg):
    """Concatenate conversation prefix up to turn_idx (0-based)."""
    parts = []
    if cfg["data"]["use_initial_contact"]:
        parts.append(record["initial_contact"])
    for i in range(turn_idx + 1):
        turn = record["engagement"][i]
        turn_parts = []
        if cfg["data"]["use_scammer_action"] and turn.get("scammer_action"):
            turn_parts.append(turn["scammer_action"])
        if cfg["data"]["use_victim_action"] and turn.get("victim_action"):
            turn_parts.append(turn["victim_action"])
        if turn_parts:
            parts.append(f"[Turn {i + 1}] " + " ".join(turn_parts))
    return " ".join(parts)


class ScamDataset(Dataset):
    """One binary example per conversation prefix.

    label = 1 if the prefix already reaches the LLM-labelled boundary turn,
    else 0.
    """

    def __init__(self, path, tokenizer, cfg):
        with open(path) as f:
            records = json.load(f)

        llm_labels = load_llm_labels(cfg["data"][f"{_split_name(path)}_labels_path"])

        self.samples = []
        for record in records:
            scam_id = record["scam_id"]
            if scam_id not in llm_labels:
                continue  # skip records without LLM labels
            detected_turn = llm_labels[scam_id]  # 1-based
            for turn_idx in range(len(record["engagement"])):
                turn_num = turn_idx + 1            # 1-based
                label = 1 if turn_num >= detected_turn else 0
                text = build_text(record, turn_idx, cfg)
                self.samples.append((text, label))

        self.tokenizer = tokenizer
        self.max_length = cfg["model"]["max_length"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": label,
        }


def _split_name(path):
    """Extract split name (train/val/test) from file path."""
    stem = str(path).split("/")[-1].replace(".json", "")
    for name in ("train", "val", "test"):
        if stem == name:
            return name
    raise ValueError(f"Cannot determine split name from path: {path}")
