"""Clean raw stream frames and label per-channel outcomes."""
from __future__ import annotations

import numpy as np
import pandas as pd

_NUMERIC_CANDIDATES = [
    "DURATION_SEC",
    "AVG_VIEWERS",
    "PEAK_VIEWERS",
    "FOLLOWERS_GAINED",
    "CHAT_MSG_COUNT",
    "BITS_TOTAL",
    "SUBS_GAINED",
]


class DataCleaner:
    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        df = df.copy()

        df["STREAM_START"] = pd.to_datetime(df["STREAM_START"], errors="coerce")
        df = df.dropna(subset=["STREAM_START"]).sort_values("STREAM_START")

        for col in _NUMERIC_CANDIDATES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[df["DURATION_SEC"].notna() & (df["DURATION_SEC"] > 300)]

        df["DURATION_HOURS"] = df["DURATION_SEC"] / 3600

        if {"PEAK_VIEWERS", "AVG_VIEWERS"}.issubset(df.columns):
            peak = df["PEAK_VIEWERS"].replace(0, np.nan)
            df["VIEWER_DROPOFF"] = (peak - df["AVG_VIEWERS"]) / peak

        if {"CHAT_MSG_COUNT", "AVG_VIEWERS"}.issubset(df.columns):
            avg = df["AVG_VIEWERS"].replace(0, np.nan)
            df["CHAT_PER_VIEWER"] = df["CHAT_MSG_COUNT"] / avg

        if {"BITS_TOTAL", "AVG_VIEWERS"}.issubset(df.columns):
            avg = df["AVG_VIEWERS"].replace(0, np.nan)
            df["BITS_PER_VIEWER"] = df["BITS_TOTAL"] / avg

        return df.reset_index(drop=True)


def label_outcome(df: pd.DataFrame, threshold_quantile: float = 0.5) -> pd.DataFrame:
    df = df.copy()
    df["CHANNEL_THRESHOLD"] = df.groupby("CHANNEL")["PEAK_VIEWERS"].transform(
        lambda s: s.quantile(threshold_quantile)
    )
    df["HITS_GOAL"] = (df["PEAK_VIEWERS"] >= df["CHANNEL_THRESHOLD"]).astype(int)
    return df
