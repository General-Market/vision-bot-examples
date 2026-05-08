from __future__ import annotations

import numpy as np
import pandas as pd

ROLLING_COLS = [
    "AVG_VIEWERS",
    "PEAK_VIEWERS",
    "DURATION_HOURS",
    "FOLLOWERS_GAINED",
    "CHAT_MSG_COUNT",
    "BITS_TOTAL",
    "SUBS_GAINED",
    "VIEWER_DROPOFF",
    "CHAT_PER_VIEWER",
    "BITS_PER_VIEWER",
]


class TwitchFeatureEngineer:
    def __init__(self, window: int = 10):
        self.window = window

    def compute_channel_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["CHANNEL", "STREAM_START"]).reset_index(drop=True)

        for col in ROLLING_COLS:
            if col not in df.columns:
                df[col] = np.nan

        grouped = df.groupby("CHANNEL", group_keys=False)

        for col in ROLLING_COLS:
            df[f"avg_{col}"] = grouped[col].transform(
                lambda s: s.shift(1).rolling(self.window, min_periods=3).mean()
            )

        if "HITS_GOAL" not in df.columns:
            df["HITS_GOAL"] = np.nan

        df["Form"] = grouped["HITS_GOAL"].transform(
            lambda s: s.shift(1).rolling(self.window, min_periods=3).mean()
        )

        df["Streak"] = grouped["HITS_GOAL"].transform(
            lambda s: TwitchFeatureEngineer._compute_streak(s.shift(1))
        )

        return df

    def build_stream_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["STREAM_START"] = pd.to_datetime(df["STREAM_START"])
        df["start_hour"] = df["STREAM_START"].dt.hour
        df["day_of_week"] = df["STREAM_START"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_primetime"] = (
            (df["start_hour"] >= 18) & (df["start_hour"] <= 23)
        ).astype(int)

        avg_cols = [f"avg_{c}" for c in ROLLING_COLS if f"avg_{c}" in df.columns]
        required = avg_cols + (["Form"] if "Form" in df.columns else [])
        if required:
            df = df.dropna(subset=required).reset_index(drop=True)

        return df

    @staticmethod
    def _compute_streak(hits: pd.Series) -> pd.Series:
        out = np.zeros(len(hits), dtype=float)
        run = 0
        for i, v in enumerate(hits.to_numpy()):
            if pd.isna(v):
                run = 0
                out[i] = 0
                continue
            hit = v >= 1
            if hit:
                run = run + 1 if run > 0 else 1
            else:
                run = run - 1 if run < 0 else -1
            out[i] = run
        return pd.Series(out, index=hits.index)


__all__ = ["TwitchFeatureEngineer", "ROLLING_COLS"]
