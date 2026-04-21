"""Weighted blend of multiple predictors."""
from __future__ import annotations


class EnsemblePredictor:
    name = "ensemble"

    def __init__(self, members: list[tuple[object, float]]):
        """`members` is a list of (predictor, weight) pairs.
        Weights need not sum to 1; the final score is the weighted mean."""
        if not members:
            raise ValueError("Ensemble needs at least one member.")
        total = sum(w for _, w in members)
        if total <= 0:
            raise ValueError("Ensemble weights must sum to > 0.")
        self.members = [(p, w / total) for p, w in members]

    def predict(self, markets, snapshot_by_id) -> list[float]:
        blended = [0.0] * len(markets)
        for predictor, weight in self.members:
            scores = predictor.predict(markets, snapshot_by_id)
            for i, s in enumerate(scores):
                blended[i] += weight * float(s)
        return blended
