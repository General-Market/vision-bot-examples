from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def _ml_to_prob(ml: float) -> float:
    if pd.isna(ml):
        return np.nan
    ml = float(ml)
    if ml < 0:
        return abs(ml) / (abs(ml) + 100.0)
    return 100.0 / (ml + 100.0)


def add_external_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if {"EXT_PEAK_LO", "EXT_PEAK_HI"}.issubset(df.columns):
        lo = df["EXT_PEAK_LO"].astype(float)
        hi = df["EXT_PEAK_HI"].astype(float)
        mean = (lo + hi) / 2.0
        std = ((hi - lo) / 2.0).replace(0, np.nan)
        z = (df["CHANNEL_THRESHOLD"].astype(float) - mean) / std
        df["ext_prob_yes"] = 1.0 - norm.cdf(z)
        df["ext_prob_no"] = 1.0 - df["ext_prob_yes"]

    if {"EXT_ML_YES", "EXT_ML_NO"}.issubset(df.columns):
        raw_yes = df["EXT_ML_YES"].apply(_ml_to_prob)
        raw_no = df["EXT_ML_NO"].apply(_ml_to_prob)
        total = raw_yes + raw_no
        df["norm_prob_yes"] = raw_yes / total
        df["norm_prob_no"] = raw_no / total
        df["odds_spread"] = df["norm_prob_yes"] - df["norm_prob_no"]

    return df


__all__ = ["add_external_signal_features"]
