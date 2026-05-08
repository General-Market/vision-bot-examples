from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    log_loss,
    roc_auc_score,
)


def classification_report_dict(y_true, y_pred) -> dict:
    return classification_report(
        y_true,
        y_pred,
        output_dict=True,
        target_names=["NO", "YES"],
    )


def summarize(y_true, y_proba) -> dict:
    y_true_arr = np.asarray(y_true)
    y_proba_arr = np.asarray(y_proba)
    pos = y_proba_arr[:, 1]
    preds = (pos >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true_arr, preds)),
        "log_loss": float(log_loss(y_true_arr, y_proba_arr, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true_arr, pos)),
        "auc": float(roc_auc_score(y_true_arr, pos)),
    }
