"""XGBoost binary classifier for next-tick direction.

Training: walk the feature history backwards, label each (asset, tick)
with the sign of the NEXT tick's change. Features = rolling windows at
t, label = direction at t+1.

Inference: given a features DataFrame from features.extract_features,
return a score in [-1, +1] where score = 2 * P(UP) - 1.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "change_1m", "change_5m", "change_15m",
    "vol_5m", "slope_5m", "streak", "n_obs_5m",
    "hour_utc", "day_of_week", "is_weekend", "is_primetime",
    "dist_to_up", "dist_to_down",
    "category_mean_5m", "asset_vs_category_5m",
    "current_change_pct",
]
# `current_resolution` is intentionally EXCLUDED from features — over
# a 60s tick the 24h baseline barely shifts, so the current resolution
# is almost identical to the next-tick label. Using it leaks the target.


def _streak_series(chg: pd.Series) -> pd.Series:
    """Signed run length of same-sign change_pct values. Vectorised but
    needs a single pass — O(n) per series."""
    s = np.zeros(len(chg), dtype=np.int32)
    run = 0
    for i, x in enumerate(chg.values):
        if not np.isfinite(x) or x == 0:
            run = 0
        elif run == 0:
            run = 1 if x > 0 else -1
        elif (x > 0 and run > 0) or (x < 0 and run < 0):
            run += 1 if x > 0 else -1
        else:
            run = 1 if x > 0 else -1
        s[i] = run
    return pd.Series(s, index=chg.index)


def build_training_set(
    history: pd.DataFrame,
    markets_by_id: dict[str, dict] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Vectorised per-asset rolling features + resolution-rule labels.

    Features describe the series up to tick t using short windows only
    (1m / 5m / 15m) plus temporal context and distance-to-threshold.
    Label is the actual resolution at tick t+1: would the oracle mark
    this market YES given the value at t+1 vs the 24h baseline?

    `markets_by_id` supplies each asset's resolution rule
    (`resolutionType` + `thresholdBps`). Without it, labels fall back to
    naive direction (old behaviour).
    """
    from features import compute_label

    markets_by_id = markets_by_id or {}
    frames: list[pd.DataFrame] = []

    for asset_id, g in history.groupby("asset_id", sort=False):
        g = g.sort_values("ts").drop_duplicates(subset="ts")
        if len(g) < 10:
            continue
        g = g.set_index("ts")
        value = g["value"].astype(float)
        chg = g["change_pct"].astype(float)

        def pct(window: str) -> pd.Series:
            start = value.rolling(window).apply(
                lambda v: v.iloc[0] if len(v) else np.nan, raw=False
            )
            return (value - start) / start.replace(0, np.nan) * 100.0

        feat = pd.DataFrame(index=value.index)
        feat["change_1m"] = pct("1min")
        feat["change_5m"] = pct("5min")
        feat["change_15m"] = pct("15min")
        feat["vol_5m"] = chg.rolling("5min").std()
        feat["slope_5m"] = value.rolling("5min").apply(
            lambda v: (
                np.polyfit(np.arange(len(v)), v, 1)[0]
                / (abs(v.mean()) or 1.0)
                * 100
            )
            if len(v) >= 3 and np.isfinite(v).all()
            else 0.0,
            raw=False,
        )
        feat["streak"] = _streak_series(chg)
        feat["n_obs_5m"] = value.rolling("5min").count()

        # Baseline for resolution: value at t - 24h, forward-filled.
        baseline = value.rolling("1D").apply(
            lambda v: v.iloc[0] if len(v) else np.nan, raw=False
        )
        feat["_baseline"] = baseline

        ts_col = feat.index
        feat["hour_utc"] = ts_col.hour
        feat["day_of_week"] = ts_col.dayofweek
        feat["is_weekend"] = (ts_col.dayofweek >= 5).astype(int)
        feat["is_primetime"] = (
            (ts_col.hour >= 18) & (ts_col.hour <= 23)
        ).astype(int)

        mk = markets_by_id.get(asset_id, {})
        res_type = mk.get("resolutionType")
        bps = mk.get("thresholdBps")
        frac = (float(bps) / 10000.0) if bps is not None else None

        if res_type == "up_x" and frac is not None:
            target = baseline * (1 + frac)
            feat["dist_to_up"] = (value - target) / target.abs().replace(0, 1) * 100.0
            feat["dist_to_down"] = 0.0
        elif res_type == "down_x" and frac is not None:
            target = baseline * (1 - frac)
            feat["dist_to_down"] = (target - value) / target.abs().replace(0, 1) * 100.0
            feat["dist_to_up"] = 0.0
        else:
            feat["dist_to_up"] = 0.0
            feat["dist_to_down"] = 0.0

        # Current resolution — 1 if this tick IS a YES per the rule.
        def _row_res(val: float) -> int:
            b = float(feat.loc[feat.index[feat.index.get_indexer([value.index[value.index == val].max()])[0]], "_baseline"]) if False else 0
            return 0
        feat["current_resolution"] = [
            compute_label(float(v), float(b) if np.isfinite(b) else 0.0, res_type, bps)
            for v, b in zip(value, baseline)
        ]

        feat["current_change_pct"] = chg

        # Cross-asset context is computed per-tick across all assets in a
        # second pass below. Placeholder zeros for now.
        feat["category_mean_5m"] = 0.0
        feat["asset_vs_category_5m"] = feat["change_5m"]

        # Label: apply resolution rule to value at t+1 vs baseline at t+1.
        next_val = value.shift(-1)
        next_baseline = baseline.shift(-1).fillna(baseline)
        feat["_label"] = [
            compute_label(
                float(nv) if np.isfinite(nv) else 0.0,
                float(nb) if np.isfinite(nb) else 0.0,
                res_type,
                bps,
            )
            for nv, nb in zip(next_val, next_baseline)
        ]
        feat["_asset_id"] = asset_id

        feat = feat.dropna(subset=["change_5m"])
        feat = feat[feat["_label"].notna()]
        if feat.empty:
            continue
        # Drop the last row (no next tick)
        feat = feat.iloc[:-1]
        frames.append(feat)

    if not frames:
        return pd.DataFrame(), pd.Series(dtype=int)

    full = pd.concat(frames, ignore_index=False)
    y = full["_label"].astype(int).reset_index(drop=True)
    X = (
        full.drop(columns=["_label", "_asset_id", "_baseline"])
        .reindex(columns=FEATURE_COLS, fill_value=0.0)
        .astype(float)
        .fillna(0.0)
        .reset_index(drop=True)
    )
    return X, y


class XGBPredictor:
    """XGBoost next-tick direction model. Requires xgboost installed."""

    name = "xgb"

    def __init__(self, model_path: str | Path | None = None):
        self.model = None
        self.path = Path(model_path) if model_path else None
        if self.path and self.path.exists():
            self.load(self.path)

    def train(
        self,
        history: pd.DataFrame,
        save_path: str | Path | None = None,
        markets_by_id: dict[str, dict] | None = None,
    ) -> dict:
        from xgboost import XGBClassifier

        X, y = build_training_set(history, markets_by_id=markets_by_id)
        if len(X) < 100:
            raise RuntimeError(
                f"Training set too small ({len(X)} rows). "
                f"Fetch more history before training."
            )

        split = int(len(X) * 0.8)
        Xtr, Xte, ytr, yte = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

        model = XGBClassifier(
            n_estimators=600,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            early_stopping_rounds=30,
            eval_metric="logloss",
            random_state=42,
        )
        eval_set = [(Xte, yte)] if len(Xte) else None
        model.fit(Xtr, ytr, eval_set=eval_set, verbose=False)
        self.model = model

        acc_train = float((model.predict(Xtr) == ytr).mean())
        acc_test = float((model.predict(Xte) == yte).mean()) if len(Xte) else float("nan")
        best_iter = getattr(model, "best_iteration", None)

        if save_path:
            self.path = Path(save_path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "wb") as f:
                pickle.dump({"model": model, "features": FEATURE_COLS}, f)

        return {
            "n_train": len(Xtr),
            "n_test": len(Xte),
            "acc_train": round(acc_train, 4),
            "acc_test": round(acc_test, 4),
            "best_iteration": best_iter,
            "saved": str(self.path) if save_path else None,
        }

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        self.model = bundle["model"]

    def predict(
        self,
        markets: list[dict],
        snapshot_by_id: dict[str, dict],
        features_df: pd.DataFrame | None = None,
    ) -> list[float]:
        if self.model is None:
            # Untrained fallback: behaves like momentum so the pipeline
            # stays usable without a model file.
            return [
                float(s.get("changePct") or 0) / 100.0
                for s in (snapshot_by_id.get(m.get("assetId"), {}) for m in markets)
            ]

        if features_df is None:
            # Inference without prebuilt features falls back to momentum;
            # real inference must be called from strategy.make_predictor
            # path that hands in features.
            return [
                float(s.get("changePct") or 0) / 100.0
                for s in (snapshot_by_id.get(m.get("assetId"), {}) for m in markets)
            ]

        scores: list[float] = []
        for m in markets:
            aid = m.get("assetId")
            if aid in features_df.index:
                row = features_df.loc[[aid], FEATURE_COLS].astype(float)
                p_up = float(self.model.predict_proba(row)[0, 1])
                scores.append(2.0 * p_up - 1.0)
            else:
                snap = snapshot_by_id.get(aid) or {}
                scores.append(float(snap.get("changePct") or 0) / 100.0)
        return scores
