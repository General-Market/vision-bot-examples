"""Per-category ELO over Polymarket settled markets.

Each category — politics, sports, crypto, world — carries a rolling
rating that gains when its markets close near the predicted outcome
(high `FINAL_CONFIDENCE`) and loses when they print surprise resolutions.
A category with a high rating is one whose recent priors are trustworthy;
a low rating means recent surprises and the model should weight its
signal less.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASELINE = 1500.0


class PolymarketELO:
    def __init__(self, k: int = 24, primetime_bonus: int = 0):
        self.k = k
        self.primetime_bonus = primetime_bonus  # kept for shape parity
        self.ratings: dict[str, float] = {}

    def get_rating(self, category: str) -> float:
        return self.ratings.get(category, BASELINE)

    @staticmethod
    def expected_score(ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    @staticmethod
    def margin_multiplier(confidence_pct: float, elo_diff: float) -> float:
        mov = abs(confidence_pct)
        return ((mov + 3) ** 0.8) / (7.5 + 0.006 * abs(elo_diff))

    def update(
        self,
        category: str,
        final_yes_price: float,
        confidence: float,
    ) -> None:
        rating = self.get_rating(category)
        expected = self.expected_score(rating, BASELINE)
        # A market that closes near 1.0 (decisive YES) is the category
        # "winning" against the coin-flip baseline. A close at 0.5 is a
        # tie and contributes no ELO change.
        actual = 1.0 if final_yes_price >= 0.5 else 0.0
        mov = float(confidence) * 100  # 0..100
        mult = self.margin_multiplier(mov, rating - BASELINE)
        self.ratings[category] = rating + self.k * mult * (actual - expected)

    def quarter_reset(self, regression_factor: float = 0.80) -> None:
        if not self.ratings:
            return
        mean = float(np.mean(list(self.ratings.values())))
        for cat, r in self.ratings.items():
            self.ratings[cat] = mean + (r - mean) * regression_factor

    def compute_elo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["CLOSED_TIME"] = pd.to_datetime(df["CLOSED_TIME"])
        df = df.sort_values("CLOSED_TIME").reset_index(drop=True)

        elo_category = np.zeros(len(df), dtype=float)
        elo_expected = np.zeros(len(df), dtype=float)
        current_quarter: str | None = None

        for i, row in df.iterrows():
            ts = row["CLOSED_TIME"]
            quarter = f"{ts.year}-Q{(ts.month - 1) // 3 + 1}"
            if current_quarter is not None and quarter != current_quarter:
                self.quarter_reset()
            current_quarter = quarter

            category = row.get("CATEGORY", "unknown")
            rating = self.get_rating(category)

            elo_category[i] = rating
            elo_expected[i] = self.expected_score(rating, BASELINE)

            yes_price = row.get("FINAL_YES_PRICE", np.nan)
            confidence = row.get("FINAL_CONFIDENCE", np.nan)
            if pd.notna(yes_price) and pd.notna(confidence):
                self.update(category, float(yes_price), float(confidence))

        df["elo_category"] = elo_category
        df["elo_expected_hit"] = elo_expected
        return df


# Backwards-compat alias for code that imports the twitch name.
TwitchELO = PolymarketELO


__all__ = ["PolymarketELO", "TwitchELO", "BASELINE"]
