"""Hierarchical encoder: round-level representation → Transformer risk scorer.

Architecture:
  1. Each round is encoded independently with a frozen sentence encoder
     (sentence-transformers/all-MiniLM-L6-v2 by default).
     initial_contact is prepended as "round 0".
  2. The resulting sequence of round embeddings is fed into a lightweight
     trainable Transformer encoder.
  3. Mean-pooled output → Linear(1) → sigmoid → risk score.

Key difference from flat Transformer/LSTM: the sequence unit is a *round*,
not a *word token*, preserving the natural turn structure of the data.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup

from rt_detect.methods._common import _make_pred, get_device

log = logging.getLogger(__name__)


# ── Round text formatting ─────────────────────────────────────────────────────

def _format_round(r, include_victim=True):
    parts = []
    if sa := r.get("scammer_action"):
        parts.append(f"Scammer: {sa}")
    if include_victim and (va := r.get("victim_action")):
        parts.append(f"Victim: {va}")
    return " | ".join(parts) if parts else ""


# ── Frozen sentence encoder ───────────────────────────────────────────────────

class _SentenceEncoder:
    """Thin wrapper around a HuggingFace encoder with mean-pooling."""

    def __init__(self, model_name, device):
        self.device    = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def embed_dim(self):
        return self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, texts, batch_size=256):
        all_embs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i: i + batch_size]
            enc = self.tokenizer(chunk, padding=True, truncation=True,
                                 max_length=128, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out  = self.model(**enc).last_hidden_state           # [B, L, D]
            mask = enc["attention_mask"].unsqueeze(-1).float()   # [B, L, 1]
            emb  = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # [B, D]
            all_embs.append(emb.cpu().numpy())
        return np.concatenate(all_embs, axis=0)  # [N, D]


# ── Pre-compute round embeddings ──────────────────────────────────────────────

def _encode_examples(examples, encoder, include_victim, include_initial_contact):
    """Return a list of float32 arrays, one per example, shape [n_rounds, D].

    Round 0 = initial_contact (if include_initial_contact).
    Rounds 1..t = context_rounds.
    """
    # Collect all texts with a pointer back to (example_idx, round_idx)
    all_texts, pointers = [], []
    for ex_i, ex in enumerate(examples):
        if include_initial_contact and ex.get("initial_contact"):
            all_texts.append(ex["initial_contact"])
            pointers.append((ex_i, 0))
        for r_i, r in enumerate(ex["context_rounds"]):
            all_texts.append(_format_round(r, include_victim))
            pointers.append((ex_i, r_i + (1 if include_initial_contact else 0)))

    if not all_texts:
        return [np.zeros((1, encoder.embed_dim), dtype="float32")] * len(examples)

    log.info(f"  Encoding {len(all_texts)} round texts with sentence encoder…")
    all_embs = encoder.encode(all_texts)  # [N_texts, D]

    # Reconstruct per-example sequences
    max_idx = [0] * len(examples)
    for ex_i, r_i in pointers:
        max_idx[ex_i] = max(max_idx[ex_i], r_i + 1)

    result = [np.zeros((max_idx[i], encoder.embed_dim), dtype="float32")
              for i in range(len(examples))]
    for txt_i, (ex_i, r_i) in enumerate(pointers):
        result[ex_i][r_i] = all_embs[txt_i]

    return result


# ── Dataset ───────────────────────────────────────────────────────────────────

class _HierDataset(Dataset):
    def __init__(self, round_embs, examples, label_key, max_rounds):
        self.round_embs = round_embs
        self.labels     = [ex[label_key] for ex in examples]
        self.max_rounds = max_rounds
        self.embed_dim  = round_embs[0].shape[-1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        emb = self.round_embs[idx]       # [n, D]
        n   = min(len(emb), self.max_rounds)
        padded      = np.zeros((self.max_rounds, self.embed_dim), dtype="float32")
        padded[:n]  = emb[:n]
        mask        = np.zeros(self.max_rounds, dtype="float32")
        mask[:n]    = 1.0
        return {
            "round_embs": torch.tensor(padded),
            "mask":       torch.tensor(mask),
            "label":      torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# ── Round-level Transformer ───────────────────────────────────────────────────

class _RoundTransformer(nn.Module):
    def __init__(self, embed_dim, nhead, num_layers, ff_dim, dropout):
        super().__init__()
        # Adjust nhead to evenly divide embed_dim
        while embed_dim % nhead != 0 and nhead > 1:
            nhead -= 1
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc      = nn.Linear(embed_dim, 1)

    def forward(self, round_embs, mask):
        # round_embs: [B, max_rounds, D]   mask: [B, max_rounds]
        pad_mask = (mask == 0)                          # True → ignored
        x = self.encoder(round_embs, src_key_padding_mask=pad_mask)
        mask_f = mask.unsqueeze(-1)                     # [B, R, 1]
        x = (x * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)  # [B, D]
        return torch.sigmoid(self.fc(x)).squeeze(-1)


# ── Main function ─────────────────────────────────────────────────────────────

def run_hierarchical(train_examples, test_examples, label_mode,
                     encoder_model="sentence-transformers/all-MiniLM-L6-v2",
                     nhead=4, num_layers=2, ff_dim=256, dropout=0.1,
                     epochs=10, batch_size=32, lr=1e-3,
                     max_rounds=10, device="cuda", out_dir=None,
                     include_victim=True, include_initial_contact=True):
    actual_device = get_device(device)
    label_key = "label_binary" if label_mode == "binary" else "label_regression"
    log.info(f"Hierarchical encoder on {actual_device}")

    # ── Frozen sentence encoder ───────────────────────────────
    encoder = _SentenceEncoder(encoder_model, actual_device)
    embed_dim = encoder.embed_dim

    # ── Pre-compute round embeddings ──────────────────────────
    train_embs = _encode_examples(train_examples, encoder,
                                  include_victim, include_initial_contact)
    test_embs  = _encode_examples(test_examples,  encoder,
                                  include_victim, include_initial_contact)

    # Free GPU memory used by sentence encoder
    encoder.model.cpu()
    del encoder
    if actual_device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Dataloaders ───────────────────────────────────────────
    train_loader = DataLoader(
        _HierDataset(train_embs, train_examples, label_key, max_rounds),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    test_loader = DataLoader(
        _HierDataset(test_embs, test_examples, label_key, max_rounds),
        batch_size=batch_size * 2, shuffle=False, num_workers=0,
    )

    # ── Model ─────────────────────────────────────────────────
    model = _RoundTransformer(embed_dim, nhead, num_layers,
                              ff_dim, dropout).to(actual_device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )
    loss_fn = nn.BCELoss() if label_mode == "binary" else nn.MSELoss()

    # ── Training ──────────────────────────────────────────────
    best_loss, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader,
                          desc=f"Hierarchical {epoch+1}/{epochs}", leave=False):
            embs  = batch["round_embs"].to(actual_device)
            mask  = batch["mask"].to(actual_device)
            y     = batch["label"].to(actual_device)
            optimizer.zero_grad()
            loss = loss_fn(model(embs, mask), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
        avg = running / len(train_loader)
        log.info(f"  Hierarchical epoch {epoch+1}: loss={avg:.4f}")
        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict({k: v.to(actual_device) for k, v in best_state.items()})

    if out_dir:
        ckpt = Path(out_dir) / "hierarchical_checkpoint"
        ckpt.mkdir(parents=True, exist_ok=True)
        model.cpu()
        torch.save(model.state_dict(), ckpt / "model.pt")
        model.to(actual_device)
        log.info(f"  Hierarchical checkpoint → {ckpt}")

    # ── Inference ─────────────────────────────────────────────
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Hierarchical inference", leave=False):
            embs = batch["round_embs"].to(actual_device)
            mask = batch["mask"].to(actual_device)
            scores.extend(model(embs, mask).cpu().numpy().tolist())

    return [_make_pred(ex, float(s)) for ex, s in zip(test_examples, scores)]
