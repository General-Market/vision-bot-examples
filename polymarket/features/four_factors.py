"""Market four factors for Polymarket events.

Borrows the Twitch bot's four-factors framing — basketball's idea that a
small set of orthogonal ratios captures most of the variance — and
applies it to prediction markets:

  RETENTION_RATE         volume sustained vs peak (proxy for sticky interest)
  TURNOVER_DENSITY       volume per liquidity-hour (orderbook churn rate)
  RESOLUTION_VELOCITY    speed of conviction in the final hours
  CATEGORY_LIFT          this market's volume vs category median
  ENGAGEMENT_INDEX       turnover × retention — a single composite
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_four_factors_stream(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ("VOLUME", "LIQUIDITY", "FINAL_CONFIDENCE"):
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0)

    volume = df["VOLUME"]
    liquidity = df["LIQUIDITY"].replace(0, np.nan)
    duration = df["DURATION_HOURS"].replace(0, np.nan)

    # Sticky interest: how much of the volume sat as resting liquidity
    # vs flashed in and out. Higher = more committed market.
    df["RETENTION_RATE"] = liquidity / volume.replace(0, np.nan)

    # Orderbook churn rate. Volume per liquidity-hour. High = active.
    turnover_denom = (liquidity * duration).replace(0, np.nan)
    df["TURNOVER_DENSITY"] = volume / turnover_denom

    # Speed of conviction. A market that printed FINAL_CONFIDENCE > 0.5
    # within a short duration moved decisively. Long-duration low-
    # confidence markets are noise.
    df["RESOLUTION_VELOCITY"] = df["FINAL_CONFIDENCE"] / duration

    if "CATEGORY" in df.columns:
        category_median = (
            df.groupby("CATEGORY")["VOLUME"]
            .transform("median")
            .replace(0, np.nan)
        )
        df["CATEGORY_LIFT"] = df["VOLUME"] / category_median

    df["ENGAGEMENT_INDEX"] = (
        df["RETENTION_RATE"].fillna(0) * df["TURNOVER_DENSITY"].fillna(0)
    )

    return df


__all__ = ["compute_four_factors_stream"]
