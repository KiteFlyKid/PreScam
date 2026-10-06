"""Shared utilities for all neural inference methods."""

import logging
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from rt_detect.data import format_context

log = logging.getLogger(__name__)


# ── Prediction helper ─────────────────────────────────────────────────────────

def _make_pred(example, score, **extra):
    return {
        "scam_id":          example["scam_id"],
        "round_index":      example["round_index"],
        "total_rounds":     example["total_rounds"],
        "label_binary":     example["label_binary"],
        "label_regression": example["label_regression"],
        "score":            score,
        **extra,
    }


# ── Device helper ─────────────────────────────────────────────────────────────

def get_device(device_str):
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Word-level tokenizer ──────────────────────────────────────────────────────

class SimpleTokenizer:
    """Word-level tokenizer; vocabulary built from training texts."""
    PAD_ID = 0
    UNK_ID = 1

    def __init__(self, max_vocab=20_000):
        self.max_vocab = max_vocab
        self.vocab: dict[str, int] = {}

    def fit(self, texts):
        counter = Counter(
            tok
            for text in texts
            for tok in re.findall(r"\w+", text.lower())
        )
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        for word, _ in counter.most_common(self.max_vocab - 2):
            self.vocab[word] = len(self.vocab)
        return self

    def encode(self, text, max_len=256):
        tokens = re.findall(r"\w+", text.lower())[:max_len]
        ids = [self.vocab.get(t, self.UNK_ID) for t in tokens]
        ids += [self.PAD_ID] * (max_len - len(ids))
        return ids[:max_len]

    @property
    def vocab_size(self):
        return len(self.vocab)


# ── Sequence dataset (shared by MLP-embed, LSTM, Transformer) ────────────────

class SeqDataset(Dataset):
    """Tokenises examples into fixed-length integer sequences.

    Yields:  input_ids (LongTensor), attention_mask (LongTensor), label (FloatTensor).
    """

    def __init__(self, examples, tokenizer, max_len, label_key,
                 include_victim=True, include_initial_contact=True):
        self.labels    = [ex[label_key] for ex in examples]
        self.encodings = [
            tokenizer.encode(
                format_context(ex, include_victim, include_initial_contact),
                max_len=max_len,
            )
            for ex in examples
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids = torch.tensor(self.encodings[idx], dtype=torch.long)
        return {
            "input_ids":      ids,
            "attention_mask": (ids != 0).long(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# ── Generic training loop for sequence models ─────────────────────────────────

def seq_train_eval(model, train_loader, device, label_mode, epochs, lr,
                   desc="Training", save_path=None):
    """Train any model with forward(input_ids, attention_mask) → float [batch].

    Returns the model loaded with the best (lowest train-loss) checkpoint.
    """
    loss_fn   = nn.BCELoss() if label_mode == "binary" else nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    best_loss, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"{desc} {epoch+1}/{epochs}",
                          leave=False):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y    = batch["label"].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(ids, mask), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
        avg = running / len(train_loader)
        log.info(f"  {desc} epoch {epoch+1}: loss={avg:.4f}")
        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.cpu().state_dict(), save_path)
        model.to(device)
        log.info(f"  Checkpoint saved → {save_path}")

    return model


# ── Generic inference for sequence models ────────────────────────────────────

@torch.no_grad()
def seq_predict(model, test_loader, device):
    model.eval()
    scores = []
    for batch in tqdm(test_loader, desc="Inference", leave=False):
        scores.extend(
            model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            ).cpu().numpy().tolist()
        )
    return scores
