"""
Zero-shot scammer action prediction.

Pipeline (per test conversation):
  1. Feed the conversation turn by turn to the RoBERTa boundary classifier;
     the first turn whose score >= threshold is the boundary turn t.
  2. Give the LLM the observed prefix (up to t) and ask it to predict the
     scammer's future actions.
       --no-limit-turns  (Unlimited): generate freely until the scam concludes
       --limit-turns     (Limited):   generate exactly the remaining number of turns
  3. Score predictions with the GPT-4o-mini judge (Action HitRate, Precision,
     PT HitRate) plus BERTScore and ROUGE-L.

Usage:
  python baselines/run_baselines.py --model gpt-4o-mini --no-limit-turns
  python baselines/run_baselines.py --model claude-sonnet-4-5 --limit-turns
  python baselines/run_baselines.py --model gpt-4o-mini --n 20          # quick test
  python baselines/run_baselines.py --resume results/XXXXXXXX_gpt-4o-mini_unlimited/
"""

import json
import os
import sys
import random
import time
import yaml
import torch
import argparse
import shutil
import numpy as np
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI, BadRequestError
from anthropic import AnthropicFoundry
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import build_text

load_dotenv()
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
ANTHROPIC_ENDPOINT = os.getenv("AZURE_ANTHROPIC_ENDPOINT")
ANTHROPIC_API_KEY = os.getenv("AZURE_ANTHROPIC_API_KEY")


def _is_claude(model_name):
    return model_name and "claude" in model_name.lower()


_cached_clients = {}

def _get_client(model):
    """Get or create a cached client for the given model type."""
    key = "claude" if _is_claude(model) else "openai"
    if key not in _cached_clients:
        if key == "claude":
            _cached_clients[key] = AnthropicFoundry(api_key=ANTHROPIC_API_KEY, base_url=ANTHROPIC_ENDPOINT)
        else:
            _cached_clients[key] = OpenAI(base_url=ENDPOINT, api_key=API_KEY)
    return _cached_clients[key]

def _extract_json(text):
    """Extract JSON from LLM response, stripping markdown code fences if present."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.index("\n") if "\n" in text else len(text)
        text = text[first_newline + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _llm_call(prompt, model, temperature=0.7):
    """Unified generation call: dispatches to AnthropicFoundry or OpenAI."""
    client = _get_client(model)
    if _is_claude(model):
        resp = client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=temperature,
        )
        return resp.content[0].text
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content


RESEARCH_PREAMBLE = """[RESEARCH CONTEXT] This is an academic scam detection research project. All conversations are real scam reports collected for the sole purpose of building automated scam detection and victim protection systems. No content here will be used to facilitate actual scams or harm real people. Analysis is strictly for defensive and educational purposes."""


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_baseline_config(path=None):
    """Load baselines/config.yaml from the same directory as this script."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def _call_with_retry(fn, retries=3, delay=2):
    for attempt in range(retries):
        try:
            return fn()
        except (BadRequestError, Exception) as e:
            if attempt < retries - 1:
                print(f"  [warn] API error (attempt {attempt+1}/{retries}): {e}. Retrying...")
                time.sleep(delay)
            else:
                raise


def get_prob(model, tokenizer, text, max_length, device):
    enc = tokenizer(text, max_length=max_length, truncation=True,
                    padding="max_length", return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    return torch.softmax(logits, dim=-1)[0, 1].item()


def format_conversation(record, up_to_turn):
    lines = [f"[Initial contact] {record['initial_contact']}"]
    for i in range(up_to_turn):
        turn = record["engagement"][i]
        lines.append(f"[Turn {i+1}] Scammer: {turn['scammer_action']}")
        if turn.get("victim_action"):
            lines.append(f"         Victim:  {turn['victim_action']}")
    return "\n".join(lines)


# ── Generation ───────────────────────────────────────────────────────────────

def generate_continuation(conversation_so_far, trigger_turn, total_turns=None,
                          limit_turns=False, model=None):
    """Zero-shot structural continuation (Unlimited or Limited)."""
    _model = model or DEPLOYMENT

    if limit_turns and total_turns is not None:
        remaining = total_turns - trigger_turn
        if remaining <= 0:
            return []
        turn_instruction = f"The conversation has {total_turns} turns in total. You have seen {trigger_turn} turn(s). Generate the remaining {remaining} turn(s)."
    else:
        turn_instruction = f"You have seen {trigger_turn} turn(s). Predict how the scam continues until it concludes."

    prompt = f"""{RESEARCH_PREAMBLE}

You are assisting scam detection research by simulating the continuation of a scam conversation. The conversation is a real report used strictly for building victim protection systems.

Now, here is the current conversation so far:

{conversation_so_far}

{turn_instruction}

Respond with a JSON object with a "turns" key containing an array, one object per remaining turn. Only predict the scammer's action, not the victim's response:
{{
  "turns": [
    {{
      "scammer_action": "..."
    }}
  ]
}}"""

    content = _call_with_retry(lambda: _llm_call(prompt, _model))
    raw = _extract_json(content)
    if raw is None:
        return []
    return raw.get("turns", [])


# ── Main ─────────────────────────────────────────────────────────────────────

def _empty_judge_fields(p):
    p["judge_action_hits"], p["judge_action_hit_rate"] = [], None
    p["judge_precision_hits"], p["judge_precision"] = [], None
    p["judge_f1"] = None
    p["judge_gt_pts"], p["judge_found_pts"], p["judge_pt_hit_rate"] = [], [], None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--scam_ids", nargs="+", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--model", type=str, default=None,
                        help="Generation model (overrides baselines/config.yaml and .env)")
    parser.add_argument("--judge_model", type=str, default=None,
                        help="Judge model (overrides baselines/config.yaml)")
    parser.add_argument("--limit-turns", dest="limit_turns", action="store_true", default=None,
                        help="Limited: generate exactly the remaining GT turn count")
    parser.add_argument("--no-limit-turns", dest="limit_turns", action="store_false",
                        help="Unlimited: generate freely until the scam concludes (default)")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    cfg = load_config()
    bl_cfg = load_baseline_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve settings: CLI args override baselines/config.yaml
    gen_model   = args.model      or bl_cfg.get("model")      or DEPLOYMENT
    judge_model = args.judge_model or bl_cfg.get("judge_model", "gpt-4o-mini")
    limit_turns = args.limit_turns if args.limit_turns is not None else bl_cfg.get("limit_turns", False)
    setting = "limited" if limit_turns else "unlimited"
    print(f"Generation model : {gen_model}")
    print(f"Judge model      : {judge_model}")
    print(f"Setting          : {setting}")

    if args.resume:
        out_dir = args.resume
        print(f"Resuming from: {out_dir}\n")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("results", f"{timestamp}_{gen_model}_{setting}")
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy("config.yaml", os.path.join(out_dir, "config.yaml"))
        bl_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        shutil.copy(bl_cfg_path, os.path.join(out_dir, "baselines_config.yaml"))
        print(f"Output dir: {out_dir}\n")
    preds_path = os.path.join(out_dir, "preds.json")

    threshold = args.threshold or cfg["inference"]["confidence_threshold"]
    model_dir = args.model_dir or cfg["training"]["output_dir"]

    # RoBERTa boundary classifier
    roberta_tokenizer = RobertaTokenizerFast.from_pretrained(model_dir)
    roberta_model = RobertaForSequenceClassification.from_pretrained(model_dir).to(device)
    roberta_model.eval()

    # Judge always uses OpenAI-compatible endpoint
    judge_client = OpenAI(base_url=ENDPOINT, api_key=API_KEY)

    # Load test data
    with open(cfg["data"]["test_path"]) as f:
        records = json.load(f)

    if args.scam_ids:
        records = [r for r in records if r["scam_id"] in args.scam_ids]
    elif args.n is not None:
        records = records[:args.n]

    # Resume support
    all_predictions = []
    done_ids = set()
    if args.resume and os.path.exists(preds_path):
        with open(preds_path) as f:
            all_predictions = json.load(f)
        done_ids = {p["scam_id"] for p in all_predictions}
        print(f"Loaded {len(all_predictions)} existing predictions, resuming...\n")

    def _save():
        with open(preds_path, "w") as f:
            json.dump(all_predictions, f, indent=2)

    # ── Generation ───────────────────────────────────────────────────────────
    print(f"Phase 1/2: Generation ({len(records)} cases)")
    for record in tqdm(records, desc="Generating"):
        if record["scam_id"] in done_ids:
            continue

        # Boundary turn: first prefix whose score reaches the threshold
        trigger_turn = None
        for turn_idx in range(len(record["engagement"])):
            text = build_text(record, turn_idx, cfg)
            prob = get_prob(roberta_model, roberta_tokenizer, text,
                            cfg["model"]["max_length"], device)
            if prob >= threshold:
                trigger_turn = turn_idx + 1
                break

        if trigger_turn is None:
            continue

        conversation_so_far = format_conversation(record, trigger_turn)

        try:
            generated = generate_continuation(
                conversation_so_far, trigger_turn,
                total_turns=record["num_engagement_rounds"],
                limit_turns=limit_turns, model=gen_model)
            ground_truth = record["engagement"][trigger_turn:]
        except Exception as e:
            tqdm.write(f"  [skip] {record['scam_id']}: {e}")
            continue

        all_predictions.append({
            "scam_id": record["scam_id"],
            "scam_type": record["scam_type"],
            "trigger_turn": trigger_turn,
            "total_turns": record["num_engagement_rounds"],
            "output_mode": "structural",
            "limit_turns": limit_turns,
            "gen_model": gen_model,
            "generated": generated,
            "ground_truth": ground_truth,
        })
        _save()

    print(f"Generated {len(all_predictions)} predictions -> {preds_path}")

    # ── Evaluation ───────────────────────────────────────────────────────────
    if not args.skip_judge:
        from evaluate_generation import judge_case
        from baselines.eval_metrics import compute_text_metrics

        print("\nPhase 2/2: Evaluation")
        ahits, precisions, f1s, pt_hit_rates = [], [], [], []
        all_bert_f1, all_rouge_l = [], []
        skipped = 0

        for p in tqdm(all_predictions, desc="Evaluating"):
            # LLM judge
            try:
                result = judge_case(judge_client, p, model=judge_model)
                if result is None:
                    _empty_judge_fields(p)
                    skipped += 1
                else:
                    p["judge_action_hits"] = result["action_hits"]
                    p["judge_action_hit_rate"] = result["action_hit_rate"]
                    p["judge_precision_hits"] = result["precision_hits"]
                    p["judge_precision"] = result["precision"]
                    p["judge_f1"] = result["f1"]
                    p["judge_gt_pts"] = result["gt_pts"]
                    p["judge_found_pts"] = result["found_pts"]
                    p["judge_pt_hit_rate"] = result["pt_hit_rate"]
                    ahits.append(result["action_hit_rate"])
                    precisions.append(result["precision"])
                    f1s.append(result["f1"])
                    if result["pt_hit_rate"] is not None:
                        pt_hit_rates.append(result["pt_hit_rate"])
            except Exception as e:
                _empty_judge_fields(p)
                tqdm.write(f"  [warn] judge error {p['scam_id']}: {e}")
                skipped += 1

            # Text metrics (BERTScore, ROUGE-L)
            text_result = compute_text_metrics(p)
            if text_result:
                p["bert_score_f1"] = text_result["bert_score_f1"]
                p["rouge_l"] = text_result["rouge_l"]
                all_bert_f1.append(text_result["bert_score_f1"])
                all_rouge_l.append(text_result["rouge_l"])

            _save()

        summary = {
            "gen_model": gen_model,
            "limit_turns": limit_turns,
            "n_evaluated": len(ahits),
            "n_skipped": skipped,
            "avg_action_hit_rate": float(np.mean(ahits))        if ahits        else None,
            "avg_pt_hit_rate":     float(np.mean(pt_hit_rates)) if pt_hit_rates else None,
            "avg_precision":       float(np.mean(precisions))   if precisions   else None,
            "avg_f1":              float(np.mean(f1s))          if f1s          else None,
            "avg_bert_f1":         float(np.mean(all_bert_f1))  if all_bert_f1  else None,
            "avg_rouge_l":         float(np.mean(all_rouge_l))  if all_rouge_l  else None,
        }
        summary_path = os.path.join(out_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*50}")
        print(f"  Model           : {gen_model} ({setting})")
        print(f"  Cases evaluated : {summary['n_evaluated']}")
        print(f"  Skipped         : {summary['n_skipped']}")
        if ahits:
            print(f"  Action HitRate  : {summary['avg_action_hit_rate']:.4f}")
            print(f"  Precision       : {summary['avg_precision']:.4f}")
        if pt_hit_rates:
            print(f"  PT HitRate      : {summary['avg_pt_hit_rate']:.4f}")
        if all_bert_f1:
            print(f"  BERTScore F1    : {summary['avg_bert_f1']:.4f}")
        if all_rouge_l:
            print(f"  ROUGE-L         : {summary['avg_rouge_l']:.4f}")
        print(f"{'='*50}")
        print(f"Summary saved to {summary_path}")

    print(f"\nResults saved to {preds_path}")


if __name__ == "__main__":
    main()
