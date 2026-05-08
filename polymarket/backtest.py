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
    markets_by_id: dict[str, dict] | None = None,
    min_history: int = 5,
) -> dict:
    """Backtest a predictor against per-asset history using the same
    resolution rule the oracle applies.

    Label at tick i = compute_label(value[i+1], baseline[i+1], res_type, bps)
    where baseline is the value 24 h earlier (or the earliest available).
    Without `markets_by_id`, falls back to naive direction labels —
    but the two paths answer different questions.
    """
    from features import compute_label, extract_features

    markets_by_id = markets_by_id or {}

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

    flips_correct: list[int] = []
    stuck_correct: list[int] = []
    for asset_id, g in history.groupby("asset_id", sort=False):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < min_history + 2:
            continue
        for i in range(min_history, len(g) - 1):
            cut = g.iloc[: i + 1]
            now_ts = cut["ts"].iloc[-1]
            try:
                feat = extract_features(
                    cut, now=now_ts, markets_by_id=markets_by_id
                )
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

            market = markets_by_id.get(asset_id, {})
            res_type = market.get("resolutionType")
            bps = market.get("thresholdBps")

            if res_type and bps is not None and asset_id in feat.index:
                baseline = float(feat.loc[asset_id, "baseline_24h"])
                truth = compute_label(next_val, baseline, res_type, bps)
                current_yes = compute_label(this_val, baseline, res_type, bps)
            else:
                truth = 1 if next_val > this_val else 0
                current_yes = 1 if float(g["change_pct"].iloc[i]) > 0 else 0

            preds.append(pred)
            probs.append(p_up)
            truths.append(truth)

            if current_yes != truth:
                flips_correct.append(1 if pred == truth else 0)
            else:
                stuck_correct.append(1 if pred == truth else 0)

    if not preds:
        return {
            "n": 0,
            "accuracy": float("nan"),
            "log_loss": float("nan"),
            "direction_up_rate": float("nan"),
        }

    flip_acc = (
        float(np.mean(flips_correct)) if flips_correct else float("nan")
    )
    stuck_acc = (
        float(np.mean(stuck_correct)) if stuck_correct else float("nan")
    )
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
        "flip_accuracy": flip_acc,
        "stuck_accuracy": stuck_acc,
        "n_flips": len(flips_correct),
        "n_stuck": len(stuck_correct),
    }
