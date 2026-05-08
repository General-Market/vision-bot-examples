from __future__ import annotations

import json
import math
from typing import Any

import anthropic

from config.settings import config

_MODEL = "claude-sonnet-4-20250514"


def _fmt(value: Any, spec: str) -> str:
    try:
        if value is None:
            return "N/A"
        if isinstance(value, float) and math.isnan(value):
            return "N/A"
        return format(value, spec)
    except (ValueError, TypeError):
        return "N/A"


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return {}


def claude_analyze_market_context(
    market_id: str,
    question: str,
    category: str,
    category_form: dict,
) -> dict:
    """Pre-trade qualitative analyst pass on a single Polymarket event."""
    client = anthropic.Anthropic()

    avg_volume = _fmt(category_form.get("avg_VOLUME", float("nan")), ",.0f")
    avg_liq = _fmt(category_form.get("avg_LIQUIDITY", float("nan")), ",.0f")
    duration = _fmt(category_form.get("avg_DURATION_HOURS", float("nan")), ".1f")
    avg_yes = _fmt(category_form.get("avg_FINAL_YES_PRICE", float("nan")), ".2f")
    avg_conf = _fmt(category_form.get("avg_FINAL_CONFIDENCE", float("nan")), ".2f")
    form = _fmt(category_form.get("Form", float("nan")), ".1%")
    streak = category_form.get("Streak", "N/A")

    prompt = f"""You are an expert prediction-market analyst. Analyze the
upcoming Polymarket event and return ONLY JSON (no markdown, no comments)
with the following scores on a scale from 0.0 to 1.0:

Market id: {market_id}
Question: "{question}"
Category: {category}

{category} category's rolling stats over last 10 closed markets:
- Avg volume: ${avg_volume}
- Avg liquidity: ${avg_liq}
- Avg open duration (h): {duration}
- Avg final YES price: {avg_yes}
- Avg final confidence: {avg_conf}
- Form (YES rate): {form}
- Streak: {streak}

Return JSON strictly in this format:
{{
    "question_specificity": <float>,
    "category_momentum": <float>,
    "information_asymmetry": <float>,
    "newsworthy_signal": <float>,
    "controversy_signal": <float>,
    "expected_resolution_speed": <float>,
    "upset_probability": <float>,
    "yes_confidence": <float>,
    "blowout_likelihood": <float>,
    "reasoning": "<brief explanation in 1-2 sentences>"
}}"""

    message = client.messages.create(
        model=_MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    text = message.content[0].text
    return _parse_json(text)


def claude_synthesize_triple_layer(
    market_id: str,
    question: str,
    ml_proba: dict,
    ext_probs: dict,
    vision_probs: dict | None,
    divergence: dict,
    vision_pool_usdc: float | None,
) -> dict:
    client = anthropic.Anthropic()

    ml_yes = _fmt(ml_proba.get("yes", float("nan")), ".1%")
    ml_no = _fmt(ml_proba.get("no", float("nan")), ".1%")
    ext_yes = _fmt(ext_probs.get("yes", float("nan")), ".1%")
    ext_no = _fmt(ext_probs.get("no", float("nan")), ".1%")

    vision_block = ""
    if vision_probs is not None:
        v_yes = _fmt(vision_probs.get("yes", float("nan")), ".1%")
        v_no = _fmt(vision_probs.get("no", float("nan")), ".1%")
        pool = _fmt(vision_pool_usdc if vision_pool_usdc is not None else float("nan"), ",.0f")
        kl = _fmt(divergence.get("kl_div_ext_vision", float("nan")), ".4f")
        consensus = "Yes" if divergence.get("all_three_agree", 0) == 1 else "No"
        vision_block = f"""
Vision testnet (crowd intelligence, Index L3):
- YES: {v_yes}
- NO: {v_no}
- Pool: ${pool} USDC (18 dec)
KL-divergence (Polymarket vs Vision): {kl}
Source consensus: {consensus}"""

    prompt = f"""You are a senior prediction-market analyst. Synthesize three sources.

Market id: {market_id}
Question: {question}

ML model:
- YES: {ml_yes}
- NO: {ml_no}

Polymarket midprice (the crowd's prior on the original venue):
- YES: {ext_yes}
- NO: {ext_no}
{vision_block}

Return ONLY JSON:
{{
    "confidence": <float 0-1>,
    "adjusted_yes": <float 0-1>,
    "adjusted_no": <float 0-1>,
    "vision_trust_level": <float 0-1>,
    "divergence_interpretation": "<string>",
    "key_insight": "<string>",
    "expected_resolution_path": "<string>",
    "risk_factor": <float 0-1>
}}"""

    message = client.messages.create(
        model=_MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    text = message.content[0].text
    result = _parse_json(text)
    if not result:
        return {"error": "parse_failed"}
    return result


_ = config

__all__ = ["claude_analyze_market_context", "claude_synthesize_triple_layer"]
