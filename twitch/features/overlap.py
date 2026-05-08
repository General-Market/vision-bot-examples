from __future__ import annotations

import numpy as np
import pandas as pd


def compute_overlap_features(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    df = df.copy()
    df["STREAM_START"] = pd.to_datetime(df["STREAM_START"])

    if "CATEGORY" not in df.columns:
        df["overlap_streams"] = 0
        df["overlap_viewer_share"] = 1.0
        return df

    overlap_streams = np.zeros(len(df), dtype=int)
    overlap_share = np.ones(len(df), dtype=float)

    starts = df["STREAM_START"].to_numpy()
    durations = df["DURATION_HOURS"].fillna(0).to_numpy()
    categories = df["CATEGORY"].to_numpy()
    avg_viewers = df["AVG_VIEWERS"].fillna(0).to_numpy()

    ends = np.array(
        [s + np.timedelta64(int(d * 3600), "s") for s, d in zip(starts, durations)]
    )

    for i in range(len(df)):
        first_hour_end = starts[i] + np.timedelta64(3600, "s")
        cat = categories[i]

        mask = (
            (categories == cat)
            & (np.arange(len(df)) != i)
            & (starts < first_hour_end)
            & (ends > starts[i])
        )
        others_viewers = avg_viewers[mask].sum()
        count = int(mask.sum())

        overlap_streams[i] = count
        mine = avg_viewers[i]
        total = mine + others_viewers
        overlap_share[i] = (mine / total) if total > 0 else 1.0

    df["overlap_streams"] = overlap_streams
    df["overlap_viewer_share"] = overlap_share
    return df


__all__ = ["compute_overlap_features"]
