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
            # logistic(x) mapped from (0,1) to (-1,+1): 2*sig - 1 = tanh(x/2)
            scores.append(math.tanh(self.gain * x))
        return scores


REGISTRY: dict[str, Callable[[], Predictor]] = {
    "all_yes": AllYes,
    "all_no": AllNo,
    "momentum": Momentum,
    "contrarian": Contrarian,
    "logistic": LogisticChange,
}


def make_predictor(name: str) -> Predictor:
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown strategy {name!r}. Known: {sorted(REGISTRY)}"
        )
    return REGISTRY[name]()


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
