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


def claude_analyze_stream_context(
    channel: str,
    upcoming_title: str,
    category: str,
    channel_form: dict,
) -> dict:
    client = anthropic.Anthropic()

    avg_viewers = _fmt(channel_form.get("avg_AVG_VIEWERS", float("nan")), ".0f")
    peak_viewers = _fmt(channel_form.get("avg_PEAK_VIEWERS", float("nan")), ".0f")
    duration = _fmt(channel_form.get("avg_DURATION_HOURS", float("nan")), ".1f")
    chat_pv = _fmt(channel_form.get("avg_CHAT_PER_VIEWER", float("nan")), ".2f")
    bits_pv = _fmt(channel_form.get("avg_BITS_PER_VIEWER", float("nan")), ".3f")
    form = _fmt(channel_form.get("Form", float("nan")), ".1%")
    streak = channel_form.get("Streak", "N/A")

    prompt = f"""You are an expert Twitch analyst. Analyze the upcoming stream
and return ONLY JSON (no markdown, no comments) with the following scores
on a scale from 0.0 to 1.0:

Channel: {channel}
Upcoming title: "{upcoming_title}"
Category: {category}

{channel}'s rolling stats over last 10 streams:
- Avg viewers: {avg_viewers}
- Peak viewers: {peak_viewers}
- Duration (h): {duration}
- Chat per viewer: {chat_pv}
- Bits per viewer: {bits_pv}
- Form (hit rate): {form}
- Streak: {streak}

Return JSON strictly in this format:
{{
    "title_hook_strength": <float>,
    "category_momentum": <float>,
    "audience_overlap_with_core": <float>,
    "collab_or_event_signal": <float>,
    "drama_or_controversy_signal": <float>,
    "expected_retention": <float>,
    "upset_probability": <float>,
    "hit_confidence": <float>,
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
    channel: str,
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
KL-divergence (Ext vs Vision): {kl}
Source consensus: {consensus}"""

    prompt = f"""You are a senior Twitch analyst. Synthesize three sources.

Channel: {channel}
Market: {question}

ML model:
- YES: {ml_yes}
- NO: {ml_no}

External analytics:
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
    "expected_peak_viewers": <int>,
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

__all__ = ["claude_analyze_stream_context", "claude_synthesize_triple_layer"]
