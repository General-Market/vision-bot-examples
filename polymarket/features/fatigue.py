"""Per-category market 'fatigue' features.

The Twitch bot tracks per-channel rest, recent-hours-streamed, and
marathon tails. The polymarket analogue tracks per-category market
density: how many markets in this category have closed in the recent
window, and how soon they came back-to-back. A flooded category
(politics in election week) may print noisy markets; a quiet category
may print sharper ones.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd


def compute_stream_fatigue_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["CLOSED_TIME"] = pd.to_datetime(df["CLOSED_TIME"])
    df = df.sort_values("CLOSED_TIME").reset_index(drop=True)

    rest_days = np.full(len(df), 3.0)
    closes_24h = np.zeros(len(df))
    closes_72h = np.zeros(len(df))
    burst_tail = np.zeros(len(df), dtype=int)

    by_category: dict[str, list[pd.Timestamp]] = {}

    for i, row in df.iterrows():
        category = row.get("CATEGORY", "unknown")
        ts = row["CLOSED_TIME"]
        history = by_category.get(category, [])

        if history:
            prev = history[-1]
            gap_hours = (ts - prev).total_seconds() / 3600.0
            rest = max(0.0, min(14.0, gap_hours / 24.0))
            rest_days[i] = rest
            if 0 <= gap_hours <= 6.0:
                burst_tail[i] = 1

        cutoff_24 = ts - timedelta(hours=24)
        cutoff_72 = ts - timedelta(hours=72)
        c24 = sum(1 for t in history if t > cutoff_24)
        c72 = sum(1 for t in history if t > cutoff_72)
        closes_24h[i] = c24
        closes_72h[i] = c72

        history.append(ts)
        by_category[category] = history

    df["rest_days"] = rest_days
    df["closes_24h"] = closes_24h
    df["closes_72h"] = closes_72h
    df["burst_tail"] = burst_tail
    df["is_back_to_back"] = (df["rest_days"] < 1.0).astype(int)

    return df


__all__ = ["compute_stream_fatigue_features"]
