from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")


def plot_twitch_divergence(
    streams: list[dict],
    output_path: str = "twitch_vision_divergence.png",
) -> str:
    ext = np.array([s.get("ext_yes", 0.0) for s in streams], dtype=float)
    vis = np.array([s.get("vision_yes", 0.0) for s in streams], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 7))

    xs = np.linspace(0, 1, 100)
    ax.fill_between(xs, np.clip(xs - 0.03, 0, 1), np.clip(xs + 0.03, 0, 1), color="#9146FF", alpha=0.12, label="±0.03 corridor")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Parity")

    ax.scatter(ext, vis, color="#9146FF", s=45, alpha=0.85, edgecolor="white", linewidth=0.5, label="Stream")

    for s in streams:
        name = s.get("name") or s.get("stream") or s.get("channel")
        if name:
            ax.annotate(
                str(name),
                (s.get("ext_yes", 0.0), s.get("vision_yes", 0.0)),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                color="#333",
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("External YES price")
    ax.set_ylabel("Vision YES price")
    ax.set_title("Twitch ↔ Vision divergence")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_triple_layer_bar(
    market_name: str,
    external: dict,
    vision: dict,
    ml_model: dict,
    output_path: str = "twitch_triple_bar.png",
) -> str:
    labels = ["YES (hits goal)", "NO"]

    def _pair(d: dict) -> list[float]:
        yes = float(d.get("YES", d.get("yes", 0.0)))
        no = float(d.get("NO", d.get("no", 1.0 - yes)))
        return [yes, no]

    ext_vals = _pair(external)
    vis_vals = _pair(vision)
    ml_vals = _pair(ml_model)

    x = np.arange(len(labels))
    width = 0.26

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_ext = ax.bar(x - width, ext_vals, width, label="External", color="#3498db")
    bars_vis = ax.bar(x, vis_vals, width, label="Vision", color="#9146FF")
    bars_ml = ax.bar(x + width, ml_vals, width, label="ML", color="#2ecc71")

    for group in (bars_ext, bars_vis, bars_ml):
        for b in group:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                h + 0.01,
                f"{h * 100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Probability")
    ax.set_title(f"{market_name} — three-layer consensus")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
