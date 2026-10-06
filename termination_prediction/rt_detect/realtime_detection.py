"""
Real-time Scam Detection — entry point.

Usage:
  python rt_detect/realtime_detection.py
  python rt_detect/realtime_detection.py --config rt_detect/config.yaml
  python rt_detect/realtime_detection.py --config rt_detect/config.yaml --limit 20
  python rt_detect/realtime_detection.py --config rt_detect/config.yaml --methods tfidf --limit 200
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Allow both `python rt_detect/realtime_detection.py` and `python -m rt_detect.realtime_detection`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rt_detect.data import build_examples, load_data, split_data_by_scam_id  # noqa: E402
from rt_detect.metrics import evaluate  # noqa: E402
from rt_detect.methods import (  # noqa: E402
    load_api_client,
    run_llm,
    run_tfidf, save_tfidf,
    run_mlp_tfidf, run_mlp_embed,
    run_lstm,
    run_transformer,
    run_bert,
    run_position_only,
    run_hierarchical,
)
from rt_detect.report import generate_report  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Real-time scam risk scoring")
    parser.add_argument("--config",  default="rt_detect/config.yaml")
    parser.add_argument("--limit",   type=int, default=None,
                        help="Cap test-set size (quick testing)")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Override methods list, e.g. --methods tfidf bert")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load and split data only; skip all method runs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.methods:
        cfg["methods"] = args.methods

    # ── Output directory ──────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = Path(cfg["output_dir"]) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output: {out_dir}")
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # ── Data ──────────────────────────────────────────────────
    project_root = Path(__file__).parent.parent
    data = load_data(project_root / cfg["input"])
    log.info(f"Loaded {len(data)} entries")

    # ── Split conversations first, then build examples separately ─
    # Train: configured sampling_mode (default pair) — balanced, independent
    # Test:  always "all" — full trajectory per conversation for AUC/AT
    ex_kwargs = dict(
        min_rounds=cfg.get("min_rounds", 3),
        scammed_only=cfg.get("scammed_only", True),
        positive_last_k=cfg.get("positive_last_k", 1),
        random_seed=cfg.get("random_seed", 42),
    )
    train_data, test_data = split_data_by_scam_id(
        data,
        test_size=cfg.get("test_size", 0.2),
        random_seed=cfg.get("random_seed", 42),
    )
    train_exs = build_examples(train_data,
                               sampling_mode=cfg.get("sampling_mode", "pair"),
                               **ex_kwargs)
    test_exs  = build_examples(test_data,
                               sampling_mode="all",  # full trajectories for eval
                               **ex_kwargs)
    if args.limit:
        # limit by conversation count, not example count
        test_ids = list({ex["scam_id"] for ex in test_exs})[: args.limit]
        test_exs = [ex for ex in test_exs if ex["scam_id"] in set(test_ids)]
    log.info(f"Train: {len(train_exs)} examples "
             f"({len({ex['scam_id'] for ex in train_exs})} conversations, "
             f"{cfg.get('sampling_mode', 'pair')} mode)")
    log.info(f"Test:  {len(test_exs)} examples "
             f"({len({ex['scam_id'] for ex in test_exs})} conversations, "
             f"all mode)")

    if args.dry_run:
        log.info("Dry run complete — data loaded, no methods executed.")
        return

    # ── Shared settings ───────────────────────────────────────
    methods      = cfg.get("methods", ["tfidf"])
    label_mode   = cfg.get("label_mode", "binary")
    inc_victim   = cfg.get("include_victim", True)
    inc_ic       = cfg.get("include_initial_contact", True)
    scammed_only = cfg.get("scammed_only", True)

    results_list  = []
    all_preds     = []

    def _record(method, preds):
        for p in preds:
            p["method"] = method
        all_preds.extend(preds)
        m = evaluate(preds)
        log.info(f"{method}  AUC={m.get('auc')}")
        results_list.append({"method": method, "label_mode": label_mode,
                              "scammed_only": scammed_only, "metrics": m})

    # ── Position-only baseline ────────────────────────────────
    if "position_only" in methods:
        log.info(f"Running position-only ({label_mode})…")
        _record("position_only", run_position_only(
            train_exs, test_exs, label_mode,
        ))

    # ── LLM ───────────────────────────────────────────────────
    if "llm" in methods:
        log.info("Running LLM inference…")
        _record("llm", run_llm(
            load_api_client(),
            cfg.get("llm_model", "openai/gpt-4o-mini"),
            test_exs,
            max_workers=cfg.get("max_workers", 50),
            include_victim=inc_victim,
            include_initial_contact=inc_ic,
        ))

    # ── TF-IDF ────────────────────────────────────────────────
    if "tfidf" in methods:
        log.info(f"Running TF-IDF ({label_mode})…")
        preds, pipe = run_tfidf(train_exs, test_exs, label_mode,
                                include_victim=inc_victim,
                                include_initial_contact=inc_ic)
        log.info(f"TF-IDF model saved to {save_tfidf(pipe, out_dir)}")
        _record("tfidf", preds)

    # ── MLP-tfidf ─────────────────────────────────────────────
    if "mlp_tfidf" in methods:
        log.info(f"Running MLP-tfidf ({label_mode})…")
        _record("mlp_tfidf", run_mlp_tfidf(
            train_exs, test_exs, label_mode,
            hidden_dims=cfg.get("mlp_hidden_dims", [512, 256]),
            epochs=cfg.get("mlp_epochs", 10),
            batch_size=cfg.get("mlp_batch_size", 64),
            lr=cfg.get("mlp_lr", 1e-3),
            dropout=cfg.get("mlp_dropout", 0.3),
            tfidf_max_features=cfg.get("mlp_tfidf_features", 10_000),
            device=cfg.get("device", "cuda"),
            out_dir=str(out_dir),
            include_victim=inc_victim,
            include_initial_contact=inc_ic,
        ))

    # ── MLP-embed ─────────────────────────────────────────────
    if "mlp_embed" in methods:
        log.info(f"Running MLP-embed ({label_mode})…")
        _record("mlp_embed", run_mlp_embed(
            train_exs, test_exs, label_mode,
            vocab_max_size=cfg.get("vocab_max_size", 20_000),
            seq_max_len=cfg.get("seq_max_len", 256),
            embed_dim=cfg.get("mlp_embed_dim", 128),
            hidden_dims=cfg.get("mlp_hidden_dims", [256, 128]),
            epochs=cfg.get("mlp_epochs", 10),
            batch_size=cfg.get("mlp_batch_size", 64),
            lr=cfg.get("mlp_lr", 1e-3),
            dropout=cfg.get("mlp_dropout", 0.3),
            device=cfg.get("device", "cuda"),
            out_dir=str(out_dir),
            include_victim=inc_victim,
            include_initial_contact=inc_ic,
        ))

    # ── LSTM ──────────────────────────────────────────────────
    if "lstm" in methods:
        log.info(f"Running LSTM ({label_mode})…")
        _record("lstm", run_lstm(
            train_exs, test_exs, label_mode,
            vocab_max_size=cfg.get("vocab_max_size", 20_000),
            seq_max_len=cfg.get("seq_max_len", 256),
            embed_dim=cfg.get("lstm_embed_dim", 128),
            hidden_dim=cfg.get("lstm_hidden_dim", 256),
            num_layers=cfg.get("lstm_num_layers", 2),
            epochs=cfg.get("lstm_epochs", 10),
            batch_size=cfg.get("lstm_batch_size", 64),
            lr=cfg.get("lstm_lr", 1e-3),
            dropout=cfg.get("lstm_dropout", 0.3),
            device=cfg.get("device", "cuda"),
            out_dir=str(out_dir),
            include_victim=inc_victim,
            include_initial_contact=inc_ic,
        ))

    # ── Transformer ───────────────────────────────────────────
    if "transformer" in methods:
        log.info(f"Running Transformer ({label_mode})…")
        _record("transformer", run_transformer(
            train_exs, test_exs, label_mode,
            vocab_max_size=cfg.get("vocab_max_size", 20_000),
            seq_max_len=cfg.get("seq_max_len", 256),
            embed_dim=cfg.get("transformer_embed_dim", 128),
            nhead=cfg.get("transformer_nhead", 4),
            num_layers=cfg.get("transformer_num_layers", 2),
            ff_dim=cfg.get("transformer_ff_dim", 512),
            epochs=cfg.get("transformer_epochs", 10),
            batch_size=cfg.get("transformer_batch_size", 64),
            lr=cfg.get("transformer_lr", 1e-3),
            dropout=cfg.get("transformer_dropout", 0.1),
            device=cfg.get("device", "cuda"),
            out_dir=str(out_dir),
            include_victim=inc_victim,
            include_initial_contact=inc_ic,
        ))

    # ── Hierarchical encoder ──────────────────────────────────
    if "hierarchical" in methods:
        log.info(f"Running Hierarchical ({label_mode})…")
        _record("hierarchical", run_hierarchical(
            train_exs, test_exs, label_mode,
            encoder_model=cfg.get("hier_encoder_model",
                                  "sentence-transformers/all-MiniLM-L6-v2"),
            nhead=cfg.get("hier_nhead", 4),
            num_layers=cfg.get("hier_num_layers", 2),
            ff_dim=cfg.get("hier_ff_dim", 256),
            dropout=cfg.get("hier_dropout", 0.1),
            epochs=cfg.get("hier_epochs", 10),
            batch_size=cfg.get("hier_batch_size", 32),
            lr=cfg.get("hier_lr", 1e-3),
            max_rounds=cfg.get("hier_max_rounds", 10),
            device=cfg.get("device", "cuda"),
            out_dir=str(out_dir),
            include_victim=inc_victim,
            include_initial_contact=inc_ic,
        ))

    # ── BERT ──────────────────────────────────────────────────
    if "bert" in methods:
        log.info(f"Running BERT ({label_mode})…")
        _record("bert", run_bert(
            train_exs, test_exs, label_mode,
            model_name=cfg.get("bert_model", "bert-base-uncased"),
            epochs=cfg.get("bert_epochs", 5),
            batch_size=cfg.get("bert_batch_size", 16),
            lr=cfg.get("bert_lr", 2e-5),
            device=cfg.get("device", "cuda"),
            out_dir=str(out_dir),
            include_victim=inc_victim,
            include_initial_contact=inc_ic,
        ))

    # ── Save outputs ──────────────────────────────────────────
    (out_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(p) for p in all_preds) + "\n", encoding="utf-8"
    )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps({"config": cfg, "timestamp": timestamp,
                    "results": results_list}, indent=2),
        encoding="utf-8",
    )

    report_path = generate_report(summary_path)
    log.info(f"Report: {report_path}")

    # ── Print summary ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Results: {out_dir}")
    print(f"{'='*60}")
    for r in results_list:
        m = r["metrics"]
        auc  = f"{m['auc']:.3f}" if m.get("auc") is not None else "—"
        aupr = f"{m['aupr']:.3f}" if m.get("aupr") is not None else "—"
        at   = f"{m['at_fpr_mean']:.2f}" if m.get("at_fpr_mean") is not None else "—"
        print(f"  {r['method']:14s}  AUC={auc}  AUPR={aupr}  AT@FPR10%={at}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
