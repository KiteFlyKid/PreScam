"""MLP risk scorers.

Two variants:
  run_mlp_tfidf  — TF-IDF sparse features → dense MLP
  run_mlp_embed  — learned word embeddings (mean-pooled) → MLP
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from rt_detect.data import format_context
from rt_detect.methods._common import (
    _make_pred, get_device,
    SeqDataset, SimpleTokenizer, seq_train_eval, seq_predict,
)

log = logging.getLogger(__name__)


# ── Shared MLP block ──────────────────────────────────────────────────────────

def _mlp_block(input_dim, hidden_dims, dropout):
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers += [nn.Linear(prev, 1)]
    return nn.Sequential(*layers), prev


# ── Variant 1: TF-IDF features → MLP ─────────────────────────────────────────

class _TfidfMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        self.net, _ = _mlp_block(input_dim, hidden_dims, dropout)

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


class _FeatureDataset(Dataset):
    def __init__(self, X, labels):
        # X: numpy float32 array [n, features]
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"features": self.X[idx], "label": self.y[idx]}


def run_mlp_tfidf(train_examples, test_examples, label_mode,
                  hidden_dims=(512, 256), epochs=10, batch_size=64,
                  lr=1e-3, dropout=0.3, tfidf_max_features=10_000,
                  device="cuda", out_dir=None,
                  include_victim=True, include_initial_contact=True):
    actual_device = get_device(device)
    label_key = "label_binary" if label_mode == "binary" else "label_regression"

    def texts(exs):
        return [format_context(ex, include_victim, include_initial_contact)
                for ex in exs]

    tfidf   = TfidfVectorizer(max_features=tfidf_max_features,
                              ngram_range=(1, 2), sublinear_tf=True)
    X_train = tfidf.fit_transform(texts(train_examples)).toarray().astype("float32")
    X_test  = tfidf.transform(texts(test_examples)).toarray().astype("float32")
    y_train = [ex[label_key] for ex in train_examples]

    train_loader = DataLoader(
        _FeatureDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )

    model     = _TfidfMLP(X_train.shape[1], list(hidden_dims), dropout).to(actual_device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )
    loss_fn = nn.BCELoss() if label_mode == "binary" else nn.MSELoss()

    best_loss, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"MLP-tfidf {epoch+1}/{epochs}",
                          leave=False):
            x = batch["features"].to(actual_device)
            y = batch["label"].to(actual_device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
        avg = running / len(train_loader)
        log.info(f"  MLP-tfidf epoch {epoch+1}: loss={avg:.4f}")
        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict({k: v.to(actual_device) for k, v in best_state.items()})

    if out_dir:
        ckpt_dir = Path(out_dir) / "mlp_tfidf_checkpoint"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.cpu().state_dict(), ckpt_dir / "model.pt")
        import pickle
        with open(ckpt_dir / "tfidf.pkl", "wb") as f:
            pickle.dump(tfidf, f)
        model.to(actual_device)
        log.info(f"  MLP-tfidf checkpoint → {ckpt_dir}")

    # Inference
    model.eval()
    scores = []
    test_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(test_tensor), batch_size * 2):
            chunk = test_tensor[i: i + batch_size * 2].to(actual_device)
            scores.extend(model(chunk).cpu().numpy().tolist())

    return [_make_pred(ex, float(s)) for ex, s in zip(test_examples, scores)]


# ── Variant 2: Embedding mean-pool → MLP ─────────────────────────────────────

class _EmbedMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dims, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout   = nn.Dropout(dropout)
        net, _         = _mlp_block(embed_dim, hidden_dims, dropout)
        self.net       = net

    def forward(self, input_ids, attention_mask):
        x    = self.embedding(input_ids)                    # [B, L, E]
        mask = attention_mask.float().unsqueeze(-1)         # [B, L, 1]
        x    = (x * mask).sum(1) / mask.sum(1).clamp(min=1)  # [B, E]
        return torch.sigmoid(self.net(self.dropout(x))).squeeze(-1)


def run_mlp_embed(train_examples, test_examples, label_mode,
                  vocab_max_size=20_000, seq_max_len=256,
                  embed_dim=128, hidden_dims=(256, 128),
                  epochs=10, batch_size=64, lr=1e-3, dropout=0.3,
                  device="cuda", out_dir=None,
                  include_victim=True, include_initial_contact=True):
    actual_device = get_device(device)
    label_key = "label_binary" if label_mode == "binary" else "label_regression"

    train_texts = [format_context(ex, include_victim, include_initial_contact)
                   for ex in train_examples]
    tokenizer = SimpleTokenizer(max_vocab=vocab_max_size).fit(train_texts)

    train_loader = DataLoader(
        SeqDataset(train_examples, tokenizer, seq_max_len, label_key,
                   include_victim, include_initial_contact),
        batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        SeqDataset(test_examples, tokenizer, seq_max_len, label_key,
                   include_victim, include_initial_contact),
        batch_size=batch_size * 2, shuffle=False, num_workers=2, pin_memory=True,
    )

    model = _EmbedMLP(tokenizer.vocab_size, embed_dim,
                      list(hidden_dims), dropout).to(actual_device)
    save_path = (Path(out_dir) / "mlp_embed_checkpoint" / "model.pt"
                 if out_dir else None)

    model = seq_train_eval(model, train_loader, actual_device, label_mode,
                           epochs, lr, desc="MLP-embed", save_path=save_path)

    scores = seq_predict(model, test_loader, actual_device)
    return [_make_pred(ex, float(s)) for ex, s in zip(test_examples, scores)]
