"""External-signal features.

For Polymarket the external signal is Polymarket itself — the CLOB
midprice is the crowd's prior on the market. We expose it as
`ext_prob_yes` / `ext_prob_no` so the triple-layer combiner can blend
it with the ML output and the Vision pool prices.

Optional inputs:
  CLOB_MID_PRICE    Polymarket midprice at observation time (0..1)
  CLOB_BID, CLOB_ASK   top-of-book quotes for spread-adjusted confidence
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_external_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Primary path: Polymarket midprice → ext probabilities.
    if "CLOB_MID_PRICE" in df.columns:
        mid = df["CLOB_MID_PRICE"].astype(float).clip(0.001, 0.999)
        df["ext_prob_yes"] = mid
        df["ext_prob_no"] = 1.0 - mid

        # Spread-adjusted confidence: tight book = sharper prior.
        if {"CLOB_BID", "CLOB_ASK"}.issubset(df.columns):
            spread = (df["CLOB_ASK"] - df["CLOB_BID"]).clip(lower=0)
            df["ext_confidence"] = (1.0 - spread.clip(0, 1)).fillna(0.5)

    # Fallback: derive from FINAL_YES_PRICE for historical rows where we
    # don't have a midprice snapshot but we do know how the market closed.
    elif "FINAL_YES_PRICE" in df.columns:
        df["ext_prob_yes"] = df["FINAL_YES_PRICE"].astype(float).clip(0.001, 0.999)
        df["ext_prob_no"] = 1.0 - df["ext_prob_yes"]

    if {"ext_prob_yes", "ext_prob_no"}.issubset(df.columns):
        df["odds_spread"] = df["ext_prob_yes"] - df["ext_prob_no"]
        df["norm_prob_yes"] = df["ext_prob_yes"]
        df["norm_prob_no"] = df["ext_prob_no"]

    return df


__all__ = ["add_external_signal_features"]
