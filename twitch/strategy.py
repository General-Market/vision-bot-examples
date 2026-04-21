"""Numerical predictors for Vision markets. Each predictor returns one
float score per market. Positive score → UP is likelier; negative → DOWN.
`picks_from_scores` binarizes the vector with a threshold (default 0.0).

All predictors follow the same contract so the trading path stays
strategy-agnostic: snapshot comes in, a vector of picks comes out.
"""
from __future__ import annotations

from typing import Callable, Protocol


class Predictor(Protocol):
    name: str

    def predict(
        self,
        markets: list[dict],
        snapshot_by_id: dict[str, dict],
    ) -> list[float]: ...


def _change_pct(snap: dict | None) -> float:
    if not snap:
        return 0.0
    try:
        return float(snap.get("changePct") or 0.0)
    except (TypeError, ValueError):
        return 0.0


class AllYes:
    name = "all_yes"

    def predict(self, markets, snapshot_by_id):
        return [1.0] * len(markets)


class AllNo:
    name = "all_no"

    def predict(self, markets, snapshot_by_id):
        return [-1.0] * len(markets)


class Momentum:
    """Trend-follow: bet UP when the asset is already moving up.

    Score = changePct / 100. Typical range [-1, +1]. Zero when we have no
    snapshot for the assetId — the threshold then decides whether the
    absence of data counts as UP or DOWN.
    """

    name = "momentum"

    def predict(self, markets, snapshot_by_id):
        return [
            _change_pct(snapshot_by_id.get(m.get("assetId"))) / 100.0
            for m in markets
        ]


class Contrarian:
    """Fade the trend."""

    name = "contrarian"

    def predict(self, markets, snapshot_by_id):
        return [
            -_change_pct(snapshot_by_id.get(m.get("assetId"))) / 100.0
            for m in markets
        ]


class LogisticChange:
    """Minimal ML: squash changePct through a logistic centered at 0.

    score in (-1, +1), interpretable as belief-strength. Demonstrates
    the "predict a number, threshold later" pattern. Gain=0.5 means
    a ±2% change maps to ≈ ±73% confidence.
    """

    name = "logistic"

    def __init__(self, gain: float = 0.5) -> None:
        self.gain = gain

    def predict(self, markets, snapshot_by_id):
        import math

        scores = []
        for m in markets:
            x = _change_pct(snapshot_by_id.get(m.get("assetId")))
            scores.append(math.tanh(self.gain * x))
        return scores


class Rolling:
    """History-aware momentum. Weighted sum of rolling pct changes at
    5m, 15m, 1h, 6h, 24h, shaped by streak length. Expects a feature
    DataFrame indexed by asset_id (features.extract_features output).
    """

    name = "rolling"
    DEFAULT_WEIGHTS = {
        "change_5m": 0.30,
        "change_15m": 0.25,
        "change_1h": 0.20,
        "change_6h": 0.15,
        "change_24h": 0.10,
    }

    def __init__(self, features_df=None, weights: dict | None = None):
        self.features_df = features_df
        self.weights = weights or self.DEFAULT_WEIGHTS

    def predict(self, markets, snapshot_by_id):
        scores = []
        f = self.features_df
        for m in markets:
            aid = m.get("assetId")
            if f is not None and aid in f.index:
                row = f.loc[aid]
                s = sum(
                    w * float(row.get(col, 0.0)) for col, w in self.weights.items()
                )
                streak = float(row.get("streak", 0.0))
                # Streak acts as confidence multiplier but capped.
                mult = 1.0 + min(0.5, 0.1 * abs(streak)) * (1 if streak >= 0 else -1) * (1 if s >= 0 else -1)
                s *= mult
                scores.append(s / 100.0)
            else:
                scores.append(
                    _change_pct(snapshot_by_id.get(aid)) / 100.0
                )
        return scores


REGISTRY: dict[str, Callable[[], Predictor]] = {
    "all_yes": AllYes,
    "all_no": AllNo,
    "momentum": Momentum,
    "contrarian": Contrarian,
    "logistic": LogisticChange,
    "rolling": Rolling,
}

# Strategies that require history + features (not zero-arg constructible).
FEATURE_STRATEGIES = {"rolling", "xgb", "claude", "ensemble"}
ALL_STRATEGIES = sorted(set(REGISTRY) | FEATURE_STRATEGIES)


def make_predictor(name: str) -> Predictor:
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown strategy {name!r}. Known: {sorted(REGISTRY)}"
        )
    return REGISTRY[name]()


def make_predictor_with_features(
    name: str,
    features_df=None,
    xgb_model_path: str | None = None,
    claude_base: str = "rolling",
    claude_top_k: int = 20,
    ensemble_spec: list[tuple[str, float]] | None = None,
):
    """Factory for history-aware / ML / LLM predictors that can't be
    built by a zero-arg constructor."""
    if name == "rolling":
        return Rolling(features_df=features_df)
    if name == "xgb":
        from xgb_predictor import XGBPredictor

        inner = XGBPredictor(xgb_model_path)

        class _XGBShim:
            name = "xgb"

            def __init__(self, inner, features_df):
                self._inner = inner
                self.features_df = features_df

            def predict(self, markets, snapshot_by_id):
                return self._inner.predict(
                    markets, snapshot_by_id, features_df=self.features_df
                )

        return _XGBShim(inner, features_df)
    if name == "claude":
        from claude_predictor import ClaudePredictor

        base = make_predictor_with_features(
            claude_base,
            features_df=features_df,
            xgb_model_path=xgb_model_path,
        )
        return ClaudePredictor(
            base_predictor=base,
            features_df=features_df,
            top_k=claude_top_k,
        )
    if name == "ensemble":
        from ensemble import EnsemblePredictor

        spec = ensemble_spec or [("rolling", 0.5), ("xgb", 0.5)]
        members = [
            (
                make_predictor_with_features(
                    n, features_df=features_df, xgb_model_path=xgb_model_path
                ),
                w,
            )
            for n, w in spec
        ]
        return EnsemblePredictor(members)
    return make_predictor(name)


def picks_from_scores(
    scores: list[float], threshold: float = 0.0
) -> list[str]:
    """Binarize a vector of scores into UP/DOWN picks.

    score > threshold  → UP
    score <= threshold → DOWN  (ties go DOWN — conservative default)
    """
    return ["UP" if s > threshold else "DOWN" for s in scores]


def pick_summary(picks: list[str]) -> dict[str, int]:
    up = sum(1 for p in picks if p == "UP")
    return {"UP": up, "DOWN": len(picks) - up, "total": len(picks)}
