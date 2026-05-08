from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd


def compute_stream_fatigue_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["STREAM_START"] = pd.to_datetime(df["STREAM_START"])
    df = df.sort_values("STREAM_START").reset_index(drop=True)

    rest_days = np.full(len(df), 3.0)
    hours_24h = np.zeros(len(df))
    hours_72h = np.zeros(len(df))
    marathon_tail = np.zeros(len(df), dtype=int)

    by_channel: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, float]]] = {}

    for i, row in df.iterrows():
        channel = row["CHANNEL"]
        start = row["STREAM_START"]
        duration = float(row.get("DURATION_HOURS", 0) or 0)
        end = start + timedelta(hours=duration)

        history = by_channel.get(channel, [])

        if history:
            prev_start, prev_end, prev_dur = history[-1]
            gap_hours = (start - prev_end).total_seconds() / 3600.0
            rest = max(0.0, min(14.0, gap_hours / 24.0))
            rest_days[i] = rest
            if prev_dur >= 8.0 and 0 <= gap_hours <= 18.0:
                marathon_tail[i] = 1

        cutoff_24 = start - timedelta(hours=24)
        cutoff_72 = start - timedelta(hours=72)
        h24 = 0.0
        h72 = 0.0
        for _s, e, d in history:
            if e > cutoff_24:
                h24 += d
            if e > cutoff_72:
                h72 += d
        hours_24h[i] = h24
        hours_72h[i] = h72

        history.append((start, end, duration))
        by_channel[channel] = history

    df["rest_days"] = rest_days
    df["hours_24h"] = hours_24h
    df["hours_72h"] = hours_72h
    df["marathon_tail"] = marathon_tail
    df["is_back_to_back"] = (df["rest_days"] < 1.0).astype(int)

    return df


__all__ = ["compute_stream_fatigue_features"]
