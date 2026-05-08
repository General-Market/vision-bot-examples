from __future__ import annotations

import numpy as np
import pandas as pd


def compute_four_factors_stream(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ("CHAT_MSG_COUNT", "BITS_TOTAL", "SUBS_GAINED"):
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0)

    peak = df["PEAK_VIEWERS"].replace(0, np.nan)
    duration = df["DURATION_HOURS"].replace(0, np.nan)

    df["RETENTION_RATE"] = df["AVG_VIEWERS"] / peak

    chat_denom = (df["AVG_VIEWERS"] * df["DURATION_HOURS"]).replace(0, np.nan)
    df["CHAT_DENSITY"] = df["CHAT_MSG_COUNT"] / chat_denom

    df["MONETIZATION_VELOCITY"] = (
        df["BITS_TOTAL"] + df["SUBS_GAINED"] * 5
    ) / duration

    if "CATEGORY" in df.columns:
        category_median = (
            df.groupby("CATEGORY")["PEAK_VIEWERS"]
            .transform("median")
            .replace(0, np.nan)
        )
        df["CATEGORY_LIFT"] = df["PEAK_VIEWERS"] / category_median

    df["ENGAGEMENT_INDEX"] = df["RETENTION_RATE"] * df["CHAT_DENSITY"]

    return df


__all__ = ["compute_four_factors_stream"]
