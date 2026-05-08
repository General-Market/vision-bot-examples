from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

sns.set_style("whitegrid")

_PALETTE = ["#2ecc71", "#3498db", "#9146FF", "#f39c12"]


def plot_model_comparison(results: dict, output_path: str = "twitch_model_comparison.png") -> str:
    names = list(results.keys())
    accuracies = [results[n].get("accuracy", 0.0) for n in names]
    losses = [results[n].get("log_loss", 0.0) for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, max(3, 0.5 * len(names) + 2)))
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(names))]

    axes[0].barh(names, accuracies, color=colors)
    axes[0].set_xlabel("Accuracy")
    axes[0].set_title("Model accuracy")
    axes[0].set_xlim(0, 1)
    for i, v in enumerate(accuracies):
        axes[0].text(v + 0.01, i, f"{v:.3f}", va="center")

    axes[1].barh(names, losses, color=colors)
    axes[1].set_xlabel("Log loss")
    axes[1].set_title("Model log loss")
    for i, v in enumerate(losses):
        axes[1].text(v, i, f"  {v:.3f}", va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_twitch_confusion_matrix(
    y_true,
    y_pred,
    output_path: str = "twitch_confusion_matrix.png",
) -> str:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    total = cm.sum() if cm.sum() > 0 else 1
    labels = ["NO (miss)", "YES (hits goal)"]

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm[i, j] / total * 100
            annot[i, j] = f"{cm[i, j]}\n({pct:.1f}%)"

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Purples",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_feature_importance(
    model,
    feature_names,
    top_n: int = 15,
    output_path: str = "twitch_feature_importance.png",
) -> str | None:
    if not hasattr(model, "feature_importances_"):
        return None

    importances = np.asarray(model.feature_importances_)
    names = np.asarray(list(feature_names))
    order = np.argsort(importances)[::-1][:top_n]

    top_names = names[order][::-1]
    top_vals = importances[order][::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(top_vals) + 1)))
    ax.barh(top_names, top_vals, color="#9146FF")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {len(top_vals)} feature importances")
    for i, v in enumerate(top_vals):
        ax.text(v, i, f"  {v:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_twitch_calibration(
    y_true,
    y_proba,
    output_path: str = "twitch_calibration.png",
) -> str:
    y_proba_arr = np.asarray(y_proba)
    pos = y_proba_arr[:, 1]
    frac_pos, mean_pred = calibration_curve(y_true, pos, n_bins=10, strategy="uniform")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, marker="o", color="#9146FF", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
