from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, log_loss


class WalkForwardBacktest:
    def __init__(self, model, scaler, initial_train_size: int = 1000, step_size: int = 25):
        self.model = model
        self.scaler = scaler
        self.initial_train_size = initial_train_size
        self.step_size = step_size

    def run(self, X, y) -> dict:
        all_preds: list = []
        all_proba: list = []
        all_true: list = []

        n = len(X)
        for start in range(self.initial_train_size, n - self.step_size, self.step_size):
            X_train = X.iloc[:start]
            y_train = y.iloc[:start] if hasattr(y, "iloc") else y[:start]

            X_test = X.iloc[start : start + self.step_size]
            y_test = y.iloc[start : start + self.step_size] if hasattr(y, "iloc") else y[start : start + self.step_size]

            self.scaler.fit(X_train)
            X_train_s = self.scaler.transform(X_train)
            X_test_s = self.scaler.transform(X_test)

            self.model.fit(X_train_s, y_train)
            preds = self.model.predict(X_test_s)
            proba = self.model.predict_proba(X_test_s)

            all_preds.extend(preds.tolist())
            all_proba.extend(proba.tolist())
            all_true.extend(list(y_test))

        all_preds_arr = np.array(all_preds)
        all_proba_arr = np.array(all_proba)
        all_true_arr = np.array(all_true)

        acc = accuracy_score(all_true_arr, all_preds_arr)
        ll = log_loss(all_true_arr, all_proba_arr, labels=[0, 1])

        print("Walk-forward backtest complete")
        print(f"  Windows evaluated: {len(all_preds_arr) // self.step_size}")
        print(f"  Total predictions: {len(all_preds_arr)}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Log loss: {ll:.4f}")
        print()
        print(classification_report(all_true_arr, all_preds_arr, target_names=["NO", "YES"]))

        return {
            "predictions": all_preds_arr,
            "probabilities": all_proba_arr,
            "actuals": all_true_arr,
            "accuracy": acc,
            "log_loss": ll,
        }
