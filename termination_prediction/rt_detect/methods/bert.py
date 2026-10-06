"""BERT fine-tuning for risk scoring (pre-trained encoder)."""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup

from rt_detect.data import format_context
from rt_detect.methods._common import _make_pred, get_device

log = logging.getLogger(__name__)


class _BertRiskScorer(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc      = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        cls = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask
                           ).last_hidden_state[:, 0, :]
        return torch.sigmoid(self.fc(self.dropout(cls))).squeeze(-1)


def run_bert(train_examples, test_examples, label_mode,
             model_name="bert-base-uncased", epochs=5, batch_size=16,
             lr=2e-5, device="cuda", out_dir=None,
             include_victim=True, include_initial_contact=True):
    actual_device = get_device(device)
    log.info(f"BERT training on {actual_device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    label_key = "label_binary" if label_mode == "binary" else "label_regression"

    class _Dataset(Dataset):
        def __init__(self, examples):
            self.examples = examples
            self.texts = [
                format_context(ex, include_victim, include_initial_contact)
                for ex in examples
            ]

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            enc = tokenizer(self.texts[idx], max_length=512, truncation=True,
                            padding="max_length", return_tensors="pt")
            return {
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(float(self.examples[idx][label_key]),
                                      dtype=torch.float32),
            }

    train_loader = DataLoader(_Dataset(train_examples), batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    model     = _BertRiskScorer(model_name).to(actual_device)
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
        total = 0.0
        for batch in tqdm(train_loader, desc=f"BERT {epoch+1}/{epochs}",
                          leave=False):
            ids  = batch["input_ids"].to(actual_device)
            mask = batch["attention_mask"].to(actual_device)
            y    = batch["label"].to(actual_device)
            optimizer.zero_grad()
            loss = loss_fn(model(ids, mask), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item()
        avg = total / len(train_loader)
        log.info(f"  BERT epoch {epoch+1}: loss={avg:.4f}")
        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict({k: v.to(actual_device) for k, v in best_state.items()})

    if out_dir:
        ckpt = Path(out_dir) / "bert_checkpoint"
        ckpt.mkdir(parents=True, exist_ok=True)
        model.cpu()
        torch.save(model.state_dict(), ckpt / "model.pt")
        tokenizer.save_pretrained(ckpt)
        log.info(f"  BERT checkpoint → {ckpt}")
        model.to(actual_device)

    test_loader = DataLoader(_Dataset(test_examples), batch_size=batch_size * 2,
                             shuffle=False, num_workers=2, pin_memory=True)
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="BERT inference", leave=False):
            scores.extend(
                model(batch["input_ids"].to(actual_device),
                      batch["attention_mask"].to(actual_device)
                      ).cpu().numpy().tolist()
            )

    return [_make_pred(ex, float(s)) for ex, s in zip(test_examples, scores)]
