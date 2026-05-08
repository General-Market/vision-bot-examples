from __future__ import annotations

import numpy as np
import pandas as pd

# Per-category rolling stats: how the category as a whole has been
# behaving recently. Keys mirror columns produced by data/cleaner.py.
ROLLING_COLS = [
    "VOLUME",
    "LIQUIDITY",
    "DURATION_HOURS",
    "VOLUME_TO_LIQUIDITY",
    "VOLUME_PER_HOUR",
    "FINAL_YES_PRICE",
    "FINAL_CONFIDENCE",
]


class PolymarketFeatureEngineer:
    def __init__(self, window: int = 10):
        self.window = window

    def compute_category_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["CATEGORY", "CLOSED_TIME"]).reset_index(drop=True)

        for col in ROLLING_COLS:
            if col not in df.columns:
                df[col] = np.nan

        grouped = df.groupby("CATEGORY", group_keys=False)

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
            lambda s: PolymarketFeatureEngineer._compute_streak(s.shift(1))
        )

        return df

    def build_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["CLOSED_TIME"] = pd.to_datetime(df["CLOSED_TIME"])
        df["close_hour"] = df["CLOSED_TIME"].dt.hour
        df["day_of_week"] = df["CLOSED_TIME"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_primetime"] = (
            (df["close_hour"] >= 18) & (df["close_hour"] <= 23)
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


__all__ = ["PolymarketFeatureEngineer", "ROLLING_COLS"]
