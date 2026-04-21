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
    "change_5m", "change_15m", "change_1h", "change_6h", "change_24h",
    "vol_1h", "vol_24h", "slope_1h", "streak", "n_obs_24h",
    "current_change_pct",
]


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
    history: pd.DataFrame, horizon_minutes: int = 1
) -> tuple[pd.DataFrame, pd.Series]:
    """Vectorised: per-asset time-indexed rolling ops. For each observed
    tick t, features describe the series up to t; label is the direction
    of the NEXT tick's value.

    Features: change_{5m,15m,1h,6h,24h}, vol_{1h,24h}, slope_1h, streak,
    n_obs_24h, current_change_pct.
    """
    frames: list[pd.DataFrame] = []

    for asset_id, g in history.groupby("asset_id", sort=False):
        g = g.sort_values("ts").drop_duplicates(subset="ts")
        if len(g) < 10:
            continue
        g = g.set_index("ts")
        value = g["value"].astype(float)
        chg = g["change_pct"].astype(float)

        # Rolling % change: compare current value to the first value
        # observed inside the `[t - window, t]` window. Robust to
        # uneven sampling — no requirement for an exact t-window sample.
        def pct(window: str) -> pd.Series:
            start = value.rolling(window).apply(
                lambda v: v.iloc[0] if len(v) else np.nan, raw=False
            )
            return (value - start) / start.replace(0, np.nan) * 100.0

        feat = pd.DataFrame(index=value.index)
        feat["change_5m"] = pct("5min")
        feat["change_15m"] = pct("15min")
        feat["change_1h"] = pct("1h")
        feat["change_6h"] = pct("6h")
        feat["change_24h"] = pct("1D")
        feat["vol_1h"] = chg.rolling("1h").std()
        feat["vol_24h"] = chg.rolling("1D").std()
        feat["slope_1h"] = (
            value.rolling("1h").apply(
                lambda v: (
                    np.polyfit(np.arange(len(v)), v, 1)[0]
                    / (abs(v.mean()) or 1.0)
                    * 100
                )
                if len(v) >= 3 and np.isfinite(v).all()
                else 0.0,
                raw=False,
            )
        )
        feat["streak"] = _streak_series(chg)
        feat["n_obs_24h"] = value.rolling("1D").count()
        feat["current_change_pct"] = chg

        # Label: next value higher than this one.
        next_val = value.shift(-1)
        label = (next_val > value).astype(int)
        feat["_label"] = label
        feat["_asset_id"] = asset_id

        # Drop the final row (no next) and any rows lacking the minimum
        # context (require at least a 1h change).
        feat = feat.dropna(subset=["change_1h", "_label"])
        if feat.empty:
            continue
        frames.append(feat)

    if not frames:
        return pd.DataFrame(), pd.Series(dtype=int)

    full = pd.concat(frames, ignore_index=False)
    y = full["_label"].astype(int)
    X = full.drop(columns=["_label", "_asset_id"]).reindex(
        columns=FEATURE_COLS, fill_value=0.0
    ).astype(float).fillna(0.0)
    y.index = range(len(y))
    X.index = range(len(X))
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
    ) -> dict:
        from xgboost import XGBClassifier

        X, y = build_training_set(history)
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
