"""Bidirectional LSTM risk scorer."""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rt_detect.data import format_context
from rt_detect.methods._common import (
    _make_pred, get_device,
    SeqDataset, SimpleTokenizer, seq_train_eval, seq_predict,
)

log = logging.getLogger(__name__)


class _BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim * 2, 1)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)             # [B, L, E]
        _, (h, _) = self.lstm(x)                  # h: [layers*2, B, H]
        # concat last-layer forward and backward hidden states
        h = torch.cat([h[-2], h[-1]], dim=-1)     # [B, H*2]
        return torch.sigmoid(self.fc(self.dropout(h))).squeeze(-1)


def run_lstm(train_examples, test_examples, label_mode,
             vocab_max_size=20_000, seq_max_len=256,
             embed_dim=128, hidden_dim=256, num_layers=2,
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

    model = _BiLSTM(tokenizer.vocab_size, embed_dim, hidden_dim,
                    num_layers, dropout).to(actual_device)
    save_path = (Path(out_dir) / "lstm_checkpoint" / "model.pt"
                 if out_dir else None)

    model = seq_train_eval(model, train_loader, actual_device, label_mode,
                           epochs, lr, desc="LSTM", save_path=save_path)

    scores = seq_predict(model, test_loader, actual_device)
    return [_make_pred(ex, float(s)) for ex, s in zip(test_examples, scores)]
