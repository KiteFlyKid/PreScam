"""LLM-based risk scoring via OpenRouter API."""

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from rt_detect.data import format_context
from rt_detect.methods._common import _make_pred

log = logging.getLogger(__name__)

LLM_RISK_PROMPT = """\
You are analyzing a scam conversation in progress. Given the history below, \
estimate the probability (0.0 to 1.0) that the scammer's NEXT action will be \
the final termination step (e.g., demanding payment, requesting personal info \
to complete a transfer, or disappearing).

{context}

Output ONLY a JSON object: {{"risk_score": <float between 0 and 1>}}"""


def load_api_client():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def _call_llm(client, model, prompt, max_tokens=100, json_mode=False,
              max_retries=3):
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs).choices[0].message.content.strip()
        except Exception as e:
            if json_mode and "response_format" in str(e) and attempt == 0:
                kwargs.pop("response_format", None)
                continue
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


def _parse_risk_score(text):
    if text is None:
        return None
    try:
        score = float(json.loads(text).get("risk_score", -1))
        if 0.0 <= score <= 1.0:
            return score
    except Exception:
        pass
    m = re.search(r'"?risk_score"?\s*:\s*([0-9]*\.?[0-9]+)', text)
    if m:
        return max(0.0, min(1.0, float(m.group(1))))
    return None


def _run_one(client, model, example, include_victim, include_initial_contact):
    ctx = format_context(example, include_victim, include_initial_contact)
    raw = _call_llm(client, model, LLM_RISK_PROMPT.format(context=ctx),
                    json_mode=True)
    return _make_pred(example, _parse_risk_score(raw), raw_response=raw)


def run_llm(client, model, examples, max_workers=50,
            include_victim=True, include_initial_contact=True):
    predictions = [None] * len(examples)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_one, client, model, ex,
                            include_victim, include_initial_contact): i
            for i, ex in enumerate(examples)
        }
        for future in tqdm(as_completed(futures), total=len(examples),
                           desc="LLM inference"):
            idx = futures[future]
            try:
                predictions[idx] = future.result()
            except Exception as e:
                log.warning(f"LLM example {idx} failed: {e}")
    return [p for p in predictions if p is not None]
