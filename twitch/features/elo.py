from __future__ import annotations

import numpy as np
import pandas as pd

BASELINE = 1500.0


class TwitchELO:
    def __init__(self, k: int = 24, primetime_bonus: int = 80):
        self.k = k
        self.primetime_bonus = primetime_bonus
        self.ratings: dict[str, float] = {}

    def get_rating(self, channel: str) -> float:
        return self.ratings.get(channel, BASELINE)

    @staticmethod
    def expected_score(ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    @staticmethod
    def margin_multiplier(viewer_over_pct: float, elo_diff: float) -> float:
        mov = abs(viewer_over_pct)
        return ((mov + 3) ** 0.8) / (7.5 + 0.006 * abs(elo_diff))

    def update(
        self,
        channel: str,
        peak_viewers: float,
        threshold: float,
        is_primetime: bool,
    ) -> None:
        rating = self.get_rating(channel)
        effective = rating + (self.primetime_bonus if is_primetime else 0)
        expected = self.expected_score(effective, BASELINE)
        actual = 1.0 if peak_viewers >= threshold else 0.0
        mov = (peak_viewers - threshold) / max(threshold, 1) * 100
        mult = self.margin_multiplier(mov, effective - BASELINE)
        self.ratings[channel] = rating + self.k * mult * (actual - expected)

    def quarter_reset(self, regression_factor: float = 0.80) -> None:
        if not self.ratings:
            return
        mean = float(np.mean(list(self.ratings.values())))
        for ch, r in self.ratings.items():
            self.ratings[ch] = mean + (r - mean) * regression_factor

    def compute_elo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["STREAM_START"] = pd.to_datetime(df["STREAM_START"])
        df = df.sort_values("STREAM_START").reset_index(drop=True)

        elo_channel = np.zeros(len(df), dtype=float)
        elo_expected = np.zeros(len(df), dtype=float)
        current_quarter: str | None = None

        threshold_col = df["THRESHOLD"] if "THRESHOLD" in df.columns else None
        primetime_col = (
            df["is_primetime"] if "is_primetime" in df.columns else None
        )

        for i, row in df.iterrows():
            ts = row["STREAM_START"]
            quarter = f"{ts.year}-Q{(ts.month - 1) // 3 + 1}"
            if current_quarter is not None and quarter != current_quarter:
                self.quarter_reset()
            current_quarter = quarter

            channel = row["CHANNEL"]
            rating = self.get_rating(channel)
            is_pt = bool(primetime_col.iloc[i]) if primetime_col is not None else False
            effective = rating + (self.primetime_bonus if is_pt else 0)

            elo_channel[i] = rating
            elo_expected[i] = self.expected_score(effective, BASELINE)

            peak = row.get("PEAK_VIEWERS", np.nan)
            if pd.notna(peak) and threshold_col is not None:
                thr = threshold_col.iloc[i]
                if pd.notna(thr):
                    self.update(channel, float(peak), float(thr), is_pt)

        df["elo_channel"] = elo_channel
        df["elo_expected_hit"] = elo_expected
        return df


__all__ = ["TwitchELO", "BASELINE"]
