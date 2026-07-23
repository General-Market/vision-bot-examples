from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config.settings import config

_LITERAL_COLS = [
    "Form",
    "Streak",
    "close_hour",
    "day_of_week",
    "is_weekend",
    "is_primetime",
    "RETENTION_RATE",
    "TURNOVER_DENSITY",
    "RESOLUTION_VELOCITY",
    "CATEGORY_LIFT",
    "ENGAGEMENT_INDEX",
]

_PREFIXES = (
    "avg_",
    "elo_",
    "rest_",
    "closes_",
    "burst_tail",
    "is_back_to_back",
    "overlap_",
    "ext_prob_",
    "norm_prob_",
    "odds_spread",
)


def prepare_model_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    cols: list[str] = []
    for c in df.columns:
        if any(c.startswith(p) for p in _PREFIXES) or c in _LITERAL_COLS:
            cols.append(c)
    seen: set[str] = set()
    cols = [c for c in cols if not (c in seen or seen.add(c))]

    X = df[cols].apply(pd.to_numeric, errors="coerce")
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(X.median(numeric_only=True))

    y = df["HITS_GOAL"].astype(int)
    feature_cols = list(X.columns)
    return X, y, feature_cols


def train_and_evaluate(
    X: pd.DataFrame, y: pd.Series
) -> tuple[dict, dict]:
    splits = TimeSeriesSplit(n_splits=5)

    def _factories() -> dict:
        return {
            "logistic": LogisticRegression(max_iter=1000, C=0.5),
            "random_forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=10,
                random_state=42,
            ),
            "xgboost": XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss",
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.08,
                random_state=42,
            ),
        }

    accuracies: dict[str, list[float]] = {k: [] for k in _factories()}
    losses: dict[str, list[float]] = {k: [] for k in _factories()}
    fitted_models: dict = {}
    fitted_scalers: dict = {}

    for train_idx, test_idx in splits.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        models = _factories()
        for name, model in models.items():
            model.fit(X_train_s, y_train)
            proba = model.predict_proba(X_test_s)
            preds = model.predict(X_test_s)
            accuracies[name].append(accuracy_score(y_test, preds))
            losses[name].append(log_loss(y_test, proba, labels=[0, 1]))
            fitted_models[name] = model
            fitted_scalers[name] = scaler

    results: dict = {}
    for name in accuracies:
        acc_arr = np.array(accuracies[name])
        loss_arr = np.array(losses[name])
        results[name] = {
            "accuracy_mean": float(acc_arr.mean()),
            "accuracy_std": float(acc_arr.std()),
            "log_loss_mean": float(loss_arr.mean()),
            "log_loss_std": float(loss_arr.std()),
        }
        print(
            f"{name:20s} acc={results[name]['accuracy_mean']:.4f}"
            f" +/-{results[name]['accuracy_std']:.4f}"
            f" logloss={results[name]['log_loss_mean']:.4f}"
            f" +/-{results[name]['log_loss_std']:.4f}"
        )

    _ = config
    return results, fitted_models


__all__ = ["prepare_model_data", "train_and_evaluate"]
