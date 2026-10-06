import yaml
import json
import numpy as np
import torch
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from scipy.special import softmax

from dataset import ScamDataset


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits, labels,
            weight=self.class_weights.to(outputs.logits.device),
        )
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    probs = softmax(logits, axis=-1)[:, 1]
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "roc_auc": roc_auc_score(labels, probs),
    }


def main():
    cfg = load_config()
    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)

    id2label = {0: "cannot_determine", 1: "can_determine"}
    label2id = {"cannot_determine": 0, "can_determine": 1}

    tokenizer = RobertaTokenizerFast.from_pretrained(cfg["model"]["name"])
    model = RobertaForSequenceClassification.from_pretrained(
        cfg["model"]["name"],
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    train_dataset = ScamDataset(cfg["data"]["train_path"], tokenizer, cfg)
    val_dataset = ScamDataset(cfg["data"]["val_path"], tokenizer, cfg)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    training_args = TrainingArguments(
        output_dir=cfg["training"]["output_dir"],
        num_train_epochs=cfg["training"]["num_epochs"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        per_device_eval_batch_size=cfg["training"]["eval_batch_size"],
        learning_rate=cfg["training"]["learning_rate"],
        warmup_steps=cfg["training"]["warmup_steps"],
        weight_decay=cfg["training"]["weight_decay"],
        logging_steps=cfg["training"]["logging_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        seed=seed,
        report_to="none",
    )

    if cfg["training"].get("use_class_weights", False):
        n0 = sum(1 for _, l in train_dataset.samples if l == 0)
        n1 = sum(1 for _, l in train_dataset.samples if l == 1)
        w0 = (n0 + n1) / (2 * n0)
        w1 = (n0 + n1) / (2 * n1)
        class_weights = torch.tensor([w0, w1], dtype=torch.float)
        print(f"Class weights: 0={w0:.3f}, 1={w1:.3f}")
        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

    trainer.train()

    trainer.save_model(cfg["training"]["output_dir"])
    tokenizer.save_pretrained(cfg["training"]["output_dir"])

    print("Training done. Evaluating on test set...")
    test_dataset = ScamDataset(cfg["data"]["test_path"], tokenizer, cfg)
    results = trainer.evaluate(test_dataset)
    print(results)


if __name__ == "__main__":
    main()
