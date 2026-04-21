"""Walk-forward backtest: score every historical tick using only data
available up to that moment, compare to the actual next-tick direction,
report accuracy and log-loss.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd


def _score_to_prob(score: float) -> float:
    # score expected in (-1, +1); map to (0, 1) via affine shift.
    p = 0.5 + 0.5 * max(-1.0, min(1.0, score))
    return min(1 - 1e-6, max(1e-6, p))


def walk_forward(
    history: pd.DataFrame,
    predictor_factory: Callable[[], object],
    min_history: int = 5,
) -> dict:
    """Backtest a predictor against per-asset history.

    For each (asset, tick_i) with i >= min_history and i < len-1:
      features from [0..i], predict score, compare to sign(value[i+1] - value[i]).

    `predictor_factory` builds a fresh predictor per backtest — the
    caller is responsible for preloading it if it depends on a saved
    model.
    """
    from features import extract_features

    preds: list[int] = []
    probs: list[float] = []
    truths: list[int] = []
    predictor = predictor_factory()

    def _set_features(pred, feat):
        """Wire per-tick features into any predictor that has the slot.
        Feature-unaware predictors ignore the attribute."""
        if hasattr(pred, "features_df"):
            pred.features_df = feat
        if hasattr(pred, "members"):
            for member, _ in pred.members:
                _set_features(member, feat)
        if hasattr(pred, "base"):
            _set_features(pred.base, feat)

    for asset_id, g in history.groupby("asset_id", sort=False):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < min_history + 2:
            continue
        for i in range(min_history, len(g) - 1):
            cut = g.iloc[: i + 1]
            now_ts = cut["ts"].iloc[-1]
            try:
                feat = extract_features(cut, now=now_ts)
            except Exception:
                continue
            _set_features(predictor, feat)
            snap = {asset_id: {"changePct": float(g["change_pct"].iloc[i])}}
            markets = [{"assetId": asset_id}]

            try:
                scores = predictor.predict(markets, snap)
            except Exception:
                continue
            if not scores:
                continue

            s = float(scores[0])
            p_up = _score_to_prob(s)
            pred = 1 if s > 0 else 0

            next_val = float(g["value"].iloc[i + 1])
            this_val = float(g["value"].iloc[i])
            if this_val == 0 or not np.isfinite(this_val):
                continue
            truth = 1 if next_val > this_val else 0

            preds.append(pred)
            probs.append(p_up)
            truths.append(truth)

    if not preds:
        return {
            "n": 0,
            "accuracy": float("nan"),
            "log_loss": float("nan"),
            "direction_up_rate": float("nan"),
        }

    acc = float(np.mean([p == t for p, t in zip(preds, truths)]))
    ll = float(
        -np.mean(
            [
                math.log(p if t == 1 else (1 - p))
                for p, t in zip(probs, truths)
            ]
        )
    )
    return {
        "n": len(preds),
        "accuracy": acc,
        "log_loss": ll,
        "direction_up_rate": float(np.mean(truths)),
    }
