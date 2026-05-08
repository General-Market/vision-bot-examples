from __future__ import annotations

import copy

import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit


class AblationStudy:
    def __init__(self, base_model, scaler, feature_sets: dict[str, list[str]]):
        self.base_model = base_model
        self.scaler = scaler
        self.feature_sets = feature_sets

    def run(self, df, y) -> dict:
        results: dict[str, dict] = {}
        tscv = TimeSeriesSplit(n_splits=5)

        for name, features in self.feature_sets.items():
            X = df[features]
            fold_acc: list[float] = []
            fold_ll: list[float] = []

            for train_idx, test_idx in tscv.split(X):
                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_train = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
                y_test = y.iloc[test_idx] if hasattr(y, "iloc") else y[test_idx]

                scaler = copy.deepcopy(self.scaler)
                model = copy.deepcopy(self.base_model)

                scaler.fit(X_train)
                X_train_s = scaler.transform(X_train)
                X_test_s = scaler.transform(X_test)

                model.fit(X_train_s, y_train)
                preds = model.predict(X_test_s)
                proba = model.predict_proba(X_test_s)

                fold_acc.append(accuracy_score(y_test, preds))
                fold_ll.append(log_loss(y_test, proba, labels=[0, 1]))

            results[name] = {
                "accuracy": float(np.mean(fold_acc)),
                "accuracy_std": float(np.std(fold_acc)),
                "log_loss": float(np.mean(fold_ll)),
                "log_loss_std": float(np.std(fold_ll)),
            }

        return results

    @staticmethod
    def print_results(results: dict) -> None:
        if not results:
            print("No ablation results.")
            return

        baseline_name = next(iter(results.keys()))
        baseline = results[baseline_name]

        print(f"Ablation study — baseline: {baseline_name}")
        print("-" * 72)
        print(f"{'Config':<28} {'Accuracy':>12} {'±std':>8} {'LogLoss':>10} {'Δacc':>8}")
        print("-" * 72)
        for name, r in results.items():
            delta = r["accuracy"] - baseline["accuracy"]
            print(
                f"{name:<28} "
                f"{r['accuracy']:>12.4f} "
                f"{r['accuracy_std']:>8.4f} "
                f"{r['log_loss']:>10.4f} "
                f"{delta:>+8.4f}"
            )
        print("-" * 72)
