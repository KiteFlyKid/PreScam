"""positive_last_k ablation (scam-only setting).

For each k in --k-values: rebuilds train/test examples with positive_last_k=k,
trains all non-LLM methods fresh, and runs the LLM method once per slug in
--llm-models. Per-k outputs go under rt_detect/results/<ts>_k_ablation/k{k}/.
A cross-k table (AUC%, AUPR%, AT@FPR10%) is written at the run root.

By default k=1, 2, 3 are all trained. Alternatively, pass --k1-results-dir
pointing to an existing k=1 run of realtime_detection.py to reuse its
predictions.jsonl instead of re-training k=1.

Usage:
  python rt_detect/run_k_ablation.py \\
      --k-values 1 2 3 \\
      --llm-models openai/gpt-4o-mini deepseek/deepseek-v3.2

Notes:
  - "scammed_only=True, sampling_mode=pair" is preserved from cfg — the
    main-experiment setting. positive_last_k is the only variable that moves.
  - At k=2 with min_rounds=3, conversations with N=3 (the majority) lose
    their entire neg_pool in pair mode and are dropped from training; in
    test ("all" mode) all their rounds become positive. At k=3 this gets
    even more extreme. This shrinks the effective dataset and shifts the
    class balance — that's what the ablation is supposed to reveal.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rt_detect.data import build_examples, load_data, split_data_by_scam_id  # noqa: E402
from rt_detect.methods import (  # noqa: E402
    load_api_client, run_llm,
    run_tfidf, save_tfidf,
    run_mlp_tfidf, run_mlp_embed,
    run_lstm, run_transformer,
    run_bert, run_position_only, run_hierarchical,
)
from rt_detect.metrics import (  # noqa: E402
    compute_auc, compute_aupr, compute_at_metrics, compute_at_fpr_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


DEFAULT_LLM_MODELS = [
    "openai/gpt-4o-mini",
    "deepseek/deepseek-v3.2",
]


def llm_short(slug):
    """openai/gpt-4o-mini → gpt-4o-mini (for column labels)."""
    return slug.split("/", 1)[-1]


# ─── Per-method dispatch (training + inference) ──────────────────────────────

def run_one_method(name, cfg, train_exs, test_exs, out_dir, label_mode,
                   inc_victim, inc_ic, llm_client=None, llm_slug=None):
    """Returns the predictions list. For LLM, caller must pass llm_slug."""
    if name == "position_only":
        return run_position_only(train_exs, test_exs, label_mode)

    if name == "tfidf":
        preds, pipe = run_tfidf(train_exs, test_exs, label_mode,
                                include_victim=inc_victim,
                                include_initial_contact=inc_ic)
        save_tfidf(pipe, out_dir)
        return preds

    if name == "mlp_tfidf":
        return run_mlp_tfidf(
            train_exs, test_exs, label_mode,
            hidden_dims=cfg.get("mlp_hidden_dims", [512, 256]),
            epochs=cfg.get("mlp_epochs", 10),
            batch_size=cfg.get("mlp_batch_size", 64),
            lr=cfg.get("mlp_lr", 1e-3),
            dropout=cfg.get("mlp_dropout", 0.3),
            tfidf_max_features=cfg.get("mlp_tfidf_features", 10_000),
            device=cfg.get("device", "cuda"), out_dir=str(out_dir),
            include_victim=inc_victim, include_initial_contact=inc_ic,
        )

    if name == "mlp_embed":
        return run_mlp_embed(
            train_exs, test_exs, label_mode,
            vocab_max_size=cfg.get("vocab_max_size", 20_000),
            seq_max_len=cfg.get("seq_max_len", 256),
            embed_dim=cfg.get("mlp_embed_dim", 128),
            hidden_dims=cfg.get("mlp_hidden_dims", [256, 128]),
            epochs=cfg.get("mlp_epochs", 10),
            batch_size=cfg.get("mlp_batch_size", 64),
            lr=cfg.get("mlp_lr", 1e-3),
            dropout=cfg.get("mlp_dropout", 0.3),
            device=cfg.get("device", "cuda"), out_dir=str(out_dir),
            include_victim=inc_victim, include_initial_contact=inc_ic,
        )

    if name == "lstm":
        return run_lstm(
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
            device=cfg.get("device", "cuda"), out_dir=str(out_dir),
            include_victim=inc_victim, include_initial_contact=inc_ic,
        )

    if name == "transformer":
        return run_transformer(
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
            device=cfg.get("device", "cuda"), out_dir=str(out_dir),
            include_victim=inc_victim, include_initial_contact=inc_ic,
        )

    if name == "hierarchical":
        return run_hierarchical(
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
            device=cfg.get("device", "cuda"), out_dir=str(out_dir),
            include_victim=inc_victim, include_initial_contact=inc_ic,
        )

    if name == "bert":
        return run_bert(
            train_exs, test_exs, label_mode,
            model_name=cfg.get("bert_model", "bert-base-uncased"),
            epochs=cfg.get("bert_epochs", 5),
            batch_size=cfg.get("bert_batch_size", 16),
            lr=cfg.get("bert_lr", 2e-5),
            device=cfg.get("device", "cuda"), out_dir=str(out_dir),
            include_victim=inc_victim, include_initial_contact=inc_ic,
        )

    if name.startswith("llm:"):
        assert llm_client is not None and llm_slug is not None
        return run_llm(
            llm_client, llm_slug, test_exs,
            max_workers=cfg.get("max_workers", 50),
            include_victim=inc_victim, include_initial_contact=inc_ic,
        )

    raise ValueError(f"Unknown method: {name}")


# ─── Metric extraction ────────────────────────────────────────────────────────

def three_metrics(preds, at_fpr=0.10):
    """Return (auc%, aupr%, at@fpr10%) for a prediction list."""
    auc  = compute_auc(preds)
    aupr = compute_aupr(preds)
    at_block = compute_at_fpr_metrics(preds, target_fpr=at_fpr)
    return {
        "auc":      auc,
        "aupr":     aupr,
        "at_fpr10": at_block.get("at_fpr_mean"),
        "tau_fpr10": at_block.get("at_fpr_tau"),
        "n_examples": len(preds),
        "n_valid": sum(1 for p in preds if p.get("score") is not None),
    }


# ─── Cross-k report ───────────────────────────────────────────────────────────

def _pct(v, d=1): return f"{v*100:.{d}f}" if v is not None else "—"
def _fmt(v, d=2): return f"{v:.{d}f}"    if v is not None else "—"


def write_cross_k_report(out_root, k_values, all_results):
    """all_results: {k: {method: metric_dict}}"""
    lines = [
        f"# positive_last_k ablation — cross-k summary",
        f"> Generated `{datetime.now().isoformat(timespec='seconds')}`\n",
        "**Setting**: scam-only (scammed_only=True, sampling_mode=pair). "
        "Only `positive_last_k` varies; everything else is held to the main "
        "pipeline config.\n",
        "## How to read",
        "",
        "`k` is `positive_last_k`: the last k rounds of each scam conversation "
        "are labelled positive (label_binary=1); rounds 1..N−k−1 are negative. "
        "k=1 is the production setting (only the termination round is "
        "positive). As k grows, the positive rate grows (≈ k/(N−1)) and AT "
        "becomes 'rounds before the start of the termination *window*'.\n",
        "## Per-method numbers across k",
        "",
    ]

    # Collect all method names (union across k)
    all_methods = []
    for k in k_values:
        for m in all_results.get(k, {}):
            if m not in all_methods:
                all_methods.append(m)

    # AUC% table
    header = ["Method"] + [f"k={k}" for k in k_values]
    sep    = ["---"] + ["---:" for _ in k_values]

    for metric_key, title in [("auc", "AUC (%) ↑"),
                              ("aupr", "AUPR (%) ↑"),
                              ("at_fpr10", "AT@FPR10% ↑")]:
        lines.append(f"### {title}\n")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(sep) + " |")
        for m in all_methods:
            row = [m]
            for k in k_values:
                cell = all_results.get(k, {}).get(m, {}).get(metric_key)
                if metric_key == "at_fpr10":
                    row.append(_fmt(cell))
                else:
                    row.append(_pct(cell))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    (out_root / "ablation_summary.md").write_text(
        "\n".join(lines), encoding="utf-8",
    )
    log.info(f"Cross-k report → {out_root / 'ablation_summary.md'}")


def load_k1_metrics(k1_dir):
    """Recompute three metrics from k=1 predictions.jsonl for consistency."""
    pred_path = Path(k1_dir) / "predictions.jsonl"
    if not pred_path.exists():
        log.warning(f"No predictions.jsonl at {pred_path}; skipping k=1")
        return {}

    by_method = {}
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)
            method = p.get("method", "unknown")
            by_method.setdefault(method, []).append(p)

    out = {}
    for method, preds in by_method.items():
        # In the k=1 run the llm slug was logged in the config (openai/gpt-4o-mini)
        # — annotate it so it lines up with the new k=2/3 column naming.
        if method == "llm":
            cfg = json.load(open(Path(k1_dir) / "summary.json")).get("config", {})
            slug = cfg.get("llm_model", "unknown")
            method = f"llm:{llm_short(slug)}"
        out[method] = three_metrics(preds)
    log.info(f"Loaded k=1 metrics: {list(out.keys())}")
    return out


# ─── Main ─────────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="positive_last_k ablation")
    parser.add_argument("--config", default="rt_detect/config.yaml")
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--llm-models", nargs="+", default=DEFAULT_LLM_MODELS,
                        help="OpenRouter slugs to run for the LLM method")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Override cfg.methods (non-LLM list)")
    parser.add_argument("--k1-results-dir",
                        default=None,
                        help="Existing k=1 results dir to merge into cross-k report")
    parser.add_argument("--no-k1-merge", action="store_true",
                        help="Skip loading k=1 from --k1-results-dir")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap test-set conversations per k (debug)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = Path(__file__).parent.parent

    # ── Output root ──────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + "_k_ablation"
    out_root = Path(cfg["output_dir"]) / timestamp
    out_root.mkdir(parents=True, exist_ok=True)
    log.info(f"Output root: {out_root}")

    run_cfg = dict(cfg)
    run_cfg["llm_models"] = args.llm_models
    run_cfg["k_values"]   = args.k_values
    with open(out_root / "ablation_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(run_cfg, f, default_flow_style=False)

    # ── Shared data ──────────────────────────────────────────────────────
    data = load_data(project_root / cfg["input"])
    log.info(f"Loaded {len(data)} scam entries")

    # Method list — strip "llm" from non-LLM list (we expand to per-model)
    base_methods = args.methods if args.methods else cfg.get("methods", [])
    non_llm_methods = [m for m in base_methods if m != "llm"]
    include_llm = "llm" in base_methods

    label_mode = cfg.get("label_mode", "binary")
    inc_victim = cfg.get("include_victim", True)
    inc_ic     = cfg.get("include_initial_contact", True)
    at_fpr     = 0.10

    # ── LLM client (only initialised if we need it) ──────────────────────
    llm_client = load_api_client() if include_llm else None

    all_results = {}

    for k in args.k_values:
        log.info(f"\n{'='*70}\n=== k = {k}\n{'='*70}")
        k_dir = out_root / f"k{k}"
        k_dir.mkdir(parents=True, exist_ok=True)

        k_cfg = dict(cfg)
        k_cfg["positive_last_k"] = k
        with open(k_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(k_cfg, f, default_flow_style=False)

        ex_kwargs = dict(
            min_rounds=k_cfg.get("min_rounds", 3),
            scammed_only=k_cfg.get("scammed_only", True),
            positive_last_k=k,
            random_seed=k_cfg.get("random_seed", 42),
        )
        train_data, test_data = split_data_by_scam_id(
            data,
            test_size=k_cfg.get("test_size", 0.2),
            random_seed=k_cfg.get("random_seed", 42),
        )
        train_exs = build_examples(train_data,
                                   sampling_mode=k_cfg.get("sampling_mode", "pair"),
                                   **ex_kwargs)
        test_exs  = build_examples(test_data, sampling_mode="all", **ex_kwargs)
        if args.limit:
            keep = set(list({ex["scam_id"] for ex in test_exs})[: args.limit])
            test_exs = [ex for ex in test_exs if ex["scam_id"] in keep]
        log.info(f"k={k}: train={len(train_exs)} ex "
                 f"({len({ex['scam_id'] for ex in train_exs})} convs), "
                 f"test={len(test_exs)} ex "
                 f"({len({ex['scam_id'] for ex in test_exs})} convs)")

        all_preds = []
        k_results = {}

        # ── Non-LLM methods ──────────────────────────────────────────────
        for method in non_llm_methods:
            log.info(f"── k={k} {method} ─────────────────────")
            preds = run_one_method(method, k_cfg, train_exs, test_exs,
                                   k_dir, label_mode, inc_victim, inc_ic)
            for p in preds:
                p["method"] = method
            all_preds.extend(preds)
            k_results[method] = three_metrics(preds, at_fpr=at_fpr)
            log.info(
                f"  k={k} {method}  "
                f"AUC={_pct(k_results[method]['auc'])}%  "
                f"AUPR={_pct(k_results[method]['aupr'])}%  "
                f"AT@FPR10={_fmt(k_results[method]['at_fpr10'])}"
            )

        # ── LLM methods (one pass per slug) ─────────────────────────────
        if include_llm:
            for slug in args.llm_models:
                short = llm_short(slug)
                method_key = f"llm:{short}"
                log.info(f"── k={k} {method_key} ({slug}) ──────────")
                preds = run_one_method(f"llm:{short}", k_cfg, train_exs, test_exs,
                                       k_dir, label_mode, inc_victim, inc_ic,
                                       llm_client=llm_client, llm_slug=slug)
                for p in preds:
                    p["method"] = method_key
                    p["llm_slug"] = slug
                all_preds.extend(preds)
                k_results[method_key] = three_metrics(preds, at_fpr=at_fpr)
                log.info(
                    f"  k={k} {method_key}  "
                    f"AUC={_pct(k_results[method_key]['auc'])}%  "
                    f"AUPR={_pct(k_results[method_key]['aupr'])}%  "
                    f"AT@FPR10={_fmt(k_results[method_key]['at_fpr10'])}"
                )

        # ── Save per-k outputs ──────────────────────────────────────────
        (k_dir / "predictions.jsonl").write_text(
            "\n".join(json.dumps(p) for p in all_preds) + "\n",
            encoding="utf-8",
        )
        (k_dir / "summary.json").write_text(
            json.dumps({
                "config": k_cfg, "positive_last_k": k,
                "results": [{"method": m, "metrics": v}
                            for m, v in k_results.items()],
            }, indent=2),
            encoding="utf-8",
        )
        all_results[k] = k_results

    # ── Merge k=1 from existing dir ──────────────────────────────────────
    final_k_values = list(args.k_values)
    if args.k1_results_dir and not args.no_k1_merge and 1 not in final_k_values:
        k1_metrics = load_k1_metrics(args.k1_results_dir)
        if k1_metrics:
            all_results[1] = k1_metrics
            final_k_values = [1] + final_k_values
            final_k_values.sort()

    # ── Cross-k summary ─────────────────────────────────────────────────
    (out_root / "ablation_summary.json").write_text(
        json.dumps({
            "k_values": final_k_values,
            "llm_models": args.llm_models if include_llm else [],
            "results": {str(k): all_results.get(k, {}) for k in final_k_values},
        }, indent=2),
        encoding="utf-8",
    )
    write_cross_k_report(out_root, final_k_values, all_results)

    # ── Console table ───────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print(f"k-ablation: {out_root}")
    print(f"{'='*78}")
    cols = final_k_values
    print(f"{'method':<40s} | " + " | ".join(
        f"k={k} (AUC/AUPR/AT)" for k in cols))
    print("-" * (40 + 25 * len(cols)))
    methods_seen = []
    for k in cols:
        for m in all_results.get(k, {}):
            if m not in methods_seen:
                methods_seen.append(m)
    for m in methods_seen:
        cells = []
        for k in cols:
            v = all_results.get(k, {}).get(m, {})
            cells.append(f"{_pct(v.get('auc')):>5s}/{_pct(v.get('aupr')):>5s}/"
                         f"{_fmt(v.get('at_fpr10')):>5s}")
        print(f"{m:<40s} | " + " | ".join(cells))
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()
