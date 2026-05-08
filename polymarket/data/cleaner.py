"""Clean raw Polymarket market frames and label binary outcomes."""
from __future__ import annotations

import numpy as np
import pandas as pd

_NUMERIC_CANDIDATES = [
    "VOLUME",
    "LIQUIDITY",
    "FINAL_YES_PRICE",
    "OUTCOME",
]


class DataCleaner:
    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        df = df.copy()

        df["CLOSED_TIME"] = pd.to_datetime(df["CLOSED_TIME"], errors="coerce")
        df["CREATED_AT"] = pd.to_datetime(df["CREATED_AT"], errors="coerce")
        df = df.dropna(subset=["CLOSED_TIME"]).sort_values("CLOSED_TIME")

        for col in _NUMERIC_CANDIDATES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop markets with no liquidity history — they carry no signal.
        df = df[df["LIQUIDITY"].notna() & (df["LIQUIDITY"] > 0)]

        # Time the market was open in hours. Mirrors twitch's
        # DURATION_HOURS feature shape.
        df["DURATION_HOURS"] = (
            (df["CLOSED_TIME"] - df["CREATED_AT"]).dt.total_seconds() / 3600.0
        )
        df["DURATION_HOURS"] = df["DURATION_HOURS"].fillna(0).clip(lower=0)

        if {"VOLUME", "LIQUIDITY"}.issubset(df.columns):
            liq = df["LIQUIDITY"].replace(0, np.nan)
            df["VOLUME_TO_LIQUIDITY"] = df["VOLUME"] / liq

        if {"VOLUME", "DURATION_HOURS"}.issubset(df.columns):
            dur = df["DURATION_HOURS"].replace(0, np.nan)
            df["VOLUME_PER_HOUR"] = df["VOLUME"] / dur

        # Approximation of "how confident the crowd was at close." A YES
        # outcome that closed at 0.95 is more confident than one at 0.55.
        if "FINAL_YES_PRICE" in df.columns:
            df["FINAL_CONFIDENCE"] = (df["FINAL_YES_PRICE"] - 0.5).abs() * 2

        return df.reset_index(drop=True)


def label_outcome(df: pd.DataFrame, threshold_quantile: float = 0.5) -> pd.DataFrame:
    """Binary label per market.

    Polymarket markets settle to 0 or 1 directly (the OUTCOME column),
    so we don't need a per-channel threshold the way the twitch bot does.
    The `threshold_quantile` argument is retained for interface parity
    with the twitch pipeline but is unused here.
    """
    _ = threshold_quantile
    df = df.copy()
    df["HITS_GOAL"] = df["OUTCOME"].astype(int)
    return df
