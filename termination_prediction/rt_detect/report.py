"""Markdown report generation for realtime detection results."""

import json
from pathlib import Path


def _fmt(v, decimals=3):
    return f"{v:.{decimals}f}" if v is not None else "—"


def _md_table(headers, rows):
    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def generate_report(summary_path):
    """Build Markdown report from summary.json; write report.md alongside it."""
    summary_path = Path(summary_path)
    with open(summary_path, encoding="utf-8") as f:
        data = json.load(f)

    cfg     = data.get("config", {})
    results = data.get("results", [])

    lines = [
        "# Real-time Termination Prediction Report",
        f"> Generated from `{summary_path}`\n",
        "## Configuration\n",
        f"- **Input**: `{cfg.get('input', '—')}`",
        f"- **scammed_only**: {cfg.get('scammed_only', '—')}",
        f"- **min_rounds**: {cfg.get('min_rounds', '—')}",
        f"- **label_mode**: {cfg.get('label_mode', '—')}",
        f"- **methods**: {cfg.get('methods', [])}",
        f"- **llm_model**: `{cfg.get('llm_model', '—')}`",
        f"- **bert_model**: `{cfg.get('bert_model', '—')}`",
        "",
        "## Results\n",
    ]

    headers = [
        "Method", "Label Mode", "Scammed Only",
        "AUC", "AUPR", "AT@FPR10%", "n_examples",
    ]
    rows = []
    for r in results:
        m = r.get("metrics", {})
        rows.append([
            r.get("method", "—"),
            r.get("label_mode", "—"),
            str(r.get("scammed_only", "—")),
            _fmt(m.get("auc")),
            _fmt(m.get("aupr")),
            _fmt(m.get("at_fpr_mean"), 2),
            str(m.get("n_examples", "—")),
        ])

    lines.append(_md_table(headers, rows))
    report_text = "\n".join(lines)

    out_path = summary_path.parent / "report.md"
    out_path.write_text(report_text, encoding="utf-8")
    return out_path
