"""Custom (non-pretrained) Transformer encoder risk scorer."""

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


class _TransformerScorer(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, num_layers,
                 ff_dim, dropout, max_len):
        super().__init__()
        self.embedding     = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc      = nn.Linear(embed_dim, 1)

    def forward(self, input_ids, attention_mask):
        B, L = input_ids.shape
        pos  = torch.arange(L, device=input_ids.device).unsqueeze(0)  # [1, L]
        x    = self.embedding(input_ids) + self.pos_embedding(pos)     # [B, L, E]

        pad_mask = (attention_mask == 0)   # True where padding → ignored
        x = self.encoder(x, src_key_padding_mask=pad_mask)             # [B, L, E]

        # Mean-pool over non-padding positions
        mask = attention_mask.float().unsqueeze(-1)                    # [B, L, 1]
        x    = (x * mask).sum(1) / mask.sum(1).clamp(min=1)           # [B, E]
        return torch.sigmoid(self.fc(x)).squeeze(-1)


def run_transformer(train_examples, test_examples, label_mode,
                    vocab_max_size=20_000, seq_max_len=256,
                    embed_dim=128, nhead=4, num_layers=2, ff_dim=512,
                    epochs=10, batch_size=64, lr=1e-3, dropout=0.1,
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

    model = _TransformerScorer(
        tokenizer.vocab_size, embed_dim, nhead, num_layers,
        ff_dim, dropout, seq_max_len,
    ).to(actual_device)
    save_path = (Path(out_dir) / "transformer_checkpoint" / "model.pt"
                 if out_dir else None)

    model = seq_train_eval(model, train_loader, actual_device, label_mode,
                           epochs, lr, desc="Transformer", save_path=save_path)

    scores = seq_predict(model, test_loader, actual_device)
    return [_make_pred(ex, float(s)) for ex, s in zip(test_examples, scores)]
