from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config.settings import config


def build_ensemble(
    X: pd.DataFrame, y: pd.Series
) -> tuple[VotingClassifier, StandardScaler]:
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    logistic = LogisticRegression(max_iter=1000, C=0.5)
    forest = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42
    )
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss",
    )

    ensemble = VotingClassifier(
        estimators=[
            ("logistic", logistic),
            ("random_forest", forest),
            ("xgboost", xgb),
        ],
        voting="soft",
        weights=[1, 1, 2],
    )
    ensemble.fit(X_train_s, y_train)

    proba = ensemble.predict_proba(X_test_s)
    preds = ensemble.predict(X_test_s)
    print(f"ensemble accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"ensemble log_loss: {log_loss(y_test, proba, labels=[0, 1]):.4f}")
    print(
        classification_report(
            y_test, preds, target_names=["NO", "YES"], zero_division=0
        )
    )

    _ = config
    return ensemble, scaler


__all__ = ["build_ensemble"]
