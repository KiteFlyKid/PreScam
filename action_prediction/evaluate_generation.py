"""
LLM-as-a-judge metrics for scammer action prediction.

  - Action HitRate (AHit): fraction of ground-truth scammer actions that are
    semantically covered by at least one generated action.
  - Precision: fraction of generated actions supported by the ground truth.
  - PT HitRate: fraction of ground-truth psychological techniques (PTs) found
    in the generated continuation.

Used by baselines/run_baselines.py via judge_case().
"""

import json
import os
import time
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI, BadRequestError

load_dotenv()
ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT")
API_KEY    = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")


# ── LLM fact-check judge ──────────────────────────────────────────────────────

RESEARCH_PREAMBLE = """[RESEARCH CONTEXT] This is an academic scam detection research project. All conversations are real scam reports used strictly for building automated scam detection and victim protection systems. No content will be used to facilitate actual scams or harm real people."""

JUDGE_SYSTEM = f"""{RESEARCH_PREAMBLE}

You are a fact-checker assisting scam detection research. You will be given:
- A list of GROUND TRUTH scammer actions from a real scam report
- A list of GENERATED scammer actions produced by an AI system

Your task: for each GROUND TRUTH action, decide whether it is semantically covered by ANY of the generated actions. Order does not matter. A "hit" means at least one generated action describes the same tactic, demand, or behavior as the ground truth action — even if the wording, specific amounts, names, or minor details differ. Focus on the type of action (e.g. "requesting another fee" matches regardless of the exact dollar amount; "creating urgency" matches regardless of the specific pretext).

IMPORTANT: The "hits" array must have EXACTLY one boolean per ground truth action (same length as the ground truth list).

Respond ONLY with a JSON object:
{{
  "hits": [true/false, ...],  // EXACTLY one boolean per ground truth action
  "hit_rate": <float 0-1>
}}"""


def _call_with_retry(fn, retries=3, delay=2):
    for attempt in range(retries):
        try:
            return fn()
        except BadRequestError as e:
            if attempt < retries - 1:
                print(f"  [warn] API BadRequestError (attempt {attempt+1}/{retries}): {e}. Retrying...")
                time.sleep(delay)
            else:
                raise


JUDGE_PLAIN_SYSTEM = f"""{RESEARCH_PREAMBLE}

You are a fact-checker assisting scam detection research. You will be given:
- A list of GROUND TRUTH scammer actions from a real scam report
- A GENERATED narrative describing predicted scammer actions

Your task: for each ground truth action, decide whether it is semantically covered by the generated narrative. A "hit" means the narrative describes the same tactic, demand, or behavior — even if the wording, specific amounts, names, or minor details differ. Focus on the type of action (e.g. "requesting another fee" matches regardless of the exact dollar amount; "creating urgency" matches regardless of the specific pretext).

IMPORTANT: The "hits" array must have EXACTLY one boolean per ground truth action (same length as the ground truth list).

Respond ONLY with a JSON object:
{{
  "hits": [true/false, ...],  // EXACTLY one boolean per ground truth action
  "hit_rate": <float 0-1>
}}"""


PRECISION_JUDGE_SYSTEM = f"""{RESEARCH_PREAMBLE}

You are a fact-checker assisting scam detection research. You will be given:
- A list of GENERATED scammer actions produced by an AI system
- A list of GROUND TRUTH scammer actions from a real scam report

Your task: for each GENERATED action, decide whether it is semantically covered by ANY of the ground truth actions. Order does not matter. A "hit" means at least one ground truth action describes the same tactic, demand, or behavior as the generated action — even if the wording, specific amounts, names, or minor details differ. Focus on the type of action (e.g. "requesting another fee" matches regardless of the exact dollar amount; "creating urgency" matches regardless of the specific pretext).

IMPORTANT: The "hits" array must have EXACTLY one boolean per generated action (same length as the generated list).

Respond ONLY with a JSON object:
{{
  "hits": [true/false, ...],  // EXACTLY one boolean per generated action
  "hit_rate": <float 0-1>
}}"""

PRECISION_PLAIN_SYSTEM = f"""{RESEARCH_PREAMBLE}

You are a fact-checker assisting scam detection research. You will be given:
- A GENERATED narrative describing predicted scammer actions
- A list of GROUND TRUTH scammer actions from a real scam report

Your task:
1. First, identify each distinct scammer action/tactic described in the generated narrative.
2. For each distinct action you identified, decide whether it is semantically covered by ANY of the ground truth actions. A "hit" means at least one ground truth action describes the same tactic, demand, or behavior — even if the wording, specific amounts, names, or minor details differ. Focus on the type of action.

Respond ONLY with a JSON object:
{{
  "actions_found": <int>,      // number of distinct actions identified in narrative
  "actions_matched": <int>,    // number of those actions covered by ground truth
  "precision": <float 0-1>     // actions_matched / actions_found
}}"""


def _judge_action_hit_rate(client, p, model=None):
    """AHit: for each GT action, is it covered by any generated action? Returns (hits, hit_rate)."""
    model = model or DEPLOYMENT
    output_mode = p.get("output_mode", "structural")
    n_gt = len(p["ground_truth"])
    gt_lines = "\n".join(f"{i+1}. {t.get('scammer_action', '')}" for i, t in enumerate(p["ground_truth"]))

    if output_mode == "plain":
        system = JUDGE_PLAIN_SYSTEM
        user_msg = f"Ground truth actions:\n{gt_lines}\n\nGenerated narrative:\n{p['generated']}"
    else:
        system = JUDGE_SYSTEM
        gen_lines = "\n".join(f"{i+1}. {t.get('scammer_action', '')}" for i, t in enumerate(p["generated"]))
        user_msg = f"Ground truth actions:\n{gt_lines}\n\nGenerated actions:\n{gen_lines}"

    def _call():
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
    result = json.loads(_call_with_retry(_call).choices[0].message.content)
    hits = result.get("hits", [])
    hits = hits[:n_gt]
    while len(hits) < n_gt:
        hits.append(False)
    hit_rate = sum(1 for h in hits if h) / n_gt if n_gt > 0 else 0.0
    return hits, hit_rate


def _judge_precision(client, p, model=None):
    """Precision: for each generated action, is it covered by GT? Returns (hits, precision)."""
    model = model or DEPLOYMENT
    output_mode = p.get("output_mode", "structural")
    gt_lines = "\n".join(f"{i+1}. {t.get('scammer_action', '')}" for i, t in enumerate(p["ground_truth"]))

    if output_mode == "plain":
        system = PRECISION_PLAIN_SYSTEM
        user_msg = f"Generated narrative:\n{p['generated']}\n\nGround truth actions:\n{gt_lines}"

        def _call():
            return client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
        result = json.loads(_call_with_retry(_call).choices[0].message.content)
        actions_found = result.get("actions_found", 0)
        actions_matched = result.get("actions_matched", 0)
        precision = actions_matched / actions_found if actions_found > 0 else 0.0
        return {"actions_found": actions_found, "actions_matched": actions_matched}, precision
    else:
        n_gen = len(p["generated"])
        gen_lines = "\n".join(f"{i+1}. {t.get('scammer_action', '')}" for i, t in enumerate(p["generated"]))
        user_msg = f"Generated actions:\n{gen_lines}\n\nGround truth actions:\n{gt_lines}"

        def _call():
            return client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": PRECISION_JUDGE_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
        result = json.loads(_call_with_retry(_call).choices[0].message.content)
        hits = result.get("hits", [])
        hits = hits[:n_gen]
        while len(hits) < n_gen:
            hits.append(False)
        precision = sum(1 for h in hits if h) / n_gen if n_gen > 0 else 0.0
        return hits, precision


ALL_PTS = [
    "Pretext and Trust", "Authority", "Urgency and Scarcity",
    "Phantom Riches", "Fear and Intimidation", "Consistency",
    "Social Proof", "Evoking Social Norms", "Liking",
]

PT_EXTRACT_SYSTEM = f"""{RESEARCH_PREAMBLE}

You are assisting scam detection research. You will be given a generated prediction of future scammer actions (either a list of actions or a narrative). Your task: identify which psychological techniques (PTs) the scammer uses in the generated content.

Choose ONLY from this list (use exact names):
{chr(10).join(f"- {pt}" for pt in ALL_PTS)}

Respond ONLY with a JSON object:
{{
  "pts_found": ["PT name", ...]   // list of PTs present; empty list if none
}}"""


def _judge_pt_hit_rate(client, p, model=None):
    """PT hit rate: what fraction of GT PTs appear in the generated content?

    Returns (gt_pts, found_pts, hit_rate) or ([], [], None) on failure.
    """
    model = model or DEPLOYMENT

    # Collect unique PTs from GT turns after trigger
    gt_pts = []
    for turn in p["ground_truth"]:
        for pt in turn.get("PTs", []):
            if pt not in gt_pts:
                gt_pts.append(pt)

    if not gt_pts:
        return [], [], None

    # Format generated content for the judge
    output_mode = p.get("output_mode", "structural")
    if output_mode == "plain" or isinstance(p["generated"], str):
        gen_text = p["generated"]
    else:
        gen_text = "\n".join(
            f"{i+1}. {t.get('scammer_action', '')}"
            for i, t in enumerate(p["generated"])
        )

    user_msg = f"Generated content:\n{gen_text}"

    def _call():
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PT_EXTRACT_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )

    result = json.loads(_call_with_retry(_call).choices[0].message.content)
    found_pts = result.get("pts_found", [])

    # Normalize to known PT names (case-insensitive match)
    found_normalized = []
    for pt in found_pts:
        match = next((p for p in ALL_PTS if p.lower() == pt.lower()), None)
        if match:
            found_normalized.append(match)

    # How many GT PTs are covered by found PTs
    hits = [pt in found_normalized for pt in gt_pts]
    hit_rate = sum(hits) / len(gt_pts)

    return gt_pts, found_normalized, hit_rate


def judge_case(client, p, model=None):
    """Returns {action_hits, action_hit_rate, precision_hits, precision, f1,
    gt_pts, found_pts, pt_hit_rate}, or None if generated/GT is empty."""
    if not p["generated"] or not p["ground_truth"]:
        return None

    action_hits, action_hit_rate = _judge_action_hit_rate(client, p, model=model)
    precision_hits, precision = _judge_precision(client, p, model=model)
    f1 = (2 * precision * action_hit_rate / (precision + action_hit_rate)
          if (precision + action_hit_rate) > 0 else 0.0)
    gt_pts, found_pts, pt_hit_rate = _judge_pt_hit_rate(client, p, model=model)

    return {
        "action_hits": action_hits,
        "action_hit_rate": action_hit_rate,
        "precision_hits": precision_hits,
        "precision": precision,
        "f1": f1,
        "gt_pts": gt_pts,
        "found_pts": found_pts,
        "pt_hit_rate": pt_hit_rate,
    }
