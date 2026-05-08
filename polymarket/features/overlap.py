"""Concurrent-market overlap features.

Twitch overlap measures: while this stream is live, how many other
streams in the same category are competing for the same viewers? The
polymarket analogue: while this market is open, how many other markets
in the same category are open and pulling at the same liquidity pool?

Output:
  overlap_markets         count of concurrent same-category markets
  overlap_volume_share    this market's volume / sum of concurrent volumes
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_overlap_features(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    _ = top_n  # kept for shape parity with the twitch signature
    df = df.copy()
    df["CLOSED_TIME"] = pd.to_datetime(df["CLOSED_TIME"])
    df["CREATED_AT"] = pd.to_datetime(df["CREATED_AT"])

    if "CATEGORY" not in df.columns:
        df["overlap_markets"] = 0
        df["overlap_volume_share"] = 1.0
        return df

    overlap_markets = np.zeros(len(df), dtype=int)
    overlap_share = np.ones(len(df), dtype=float)

    starts = df["CREATED_AT"].to_numpy()
    ends = df["CLOSED_TIME"].to_numpy()
    categories = df["CATEGORY"].to_numpy()
    volume = df["VOLUME"].fillna(0).to_numpy()

    for i in range(len(df)):
        cat = categories[i]
        # Markets whose lifetimes intersect this one's, same category,
        # excluding self.
        mask = (
            (categories == cat)
            & (np.arange(len(df)) != i)
            & (starts < ends[i])
            & (ends > starts[i])
        )
        others_volume = volume[mask].sum()
        count = int(mask.sum())

        overlap_markets[i] = count
        mine = volume[i]
        total = mine + others_volume
        overlap_share[i] = (mine / total) if total > 0 else 1.0

    df["overlap_markets"] = overlap_markets
    df["overlap_volume_share"] = overlap_share
    return df


__all__ = ["compute_overlap_features"]
