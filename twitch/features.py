"""History-based feature extraction for Vision markets.

Per asset, builds: multi-window rolling changes, volatility, streak,
regression slope, sample count. Output is one row per asset_id — what
every history-aware predictor consumes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = [
    ("change_5m", 5),
    ("change_15m", 15),
    ("change_1h", 60),
    ("change_6h", 360),
    ("change_24h", 1440),
]


def _pct_change(now: float, then: float) -> float:
    if then == 0 or not np.isfinite(then):
        return 0.0
    return (now - then) / then * 100.0


def _streak(changes: pd.Series) -> int:
    """Length of the current same-sign run in the tail of the series.
    Positive if trending up, negative if trending down."""
    s = 0
    for x in changes.values[::-1]:
        if not np.isfinite(x) or x == 0:
            break
        if s == 0:
            s = 1 if x > 0 else -1
            continue
        if (x > 0 and s > 0) or (x < 0 and s < 0):
            s += 1 if x > 0 else -1
        else:
            break
    return s


def _slope(values: pd.Series) -> float:
    """Normalised linear-regression slope (%/hour). Robust to scale."""
    if len(values) < 3:
        return 0.0
    v = values.values.astype(float)
    if not np.all(np.isfinite(v)):
        v = v[np.isfinite(v)]
        if len(v) < 3:
            return 0.0
    x = np.arange(len(v), dtype=float)
    x -= x.mean()
    y = v - v.mean()
    denom = (x * x).sum()
    if denom == 0:
        return 0.0
    beta = (x * y).sum() / denom
    scale = abs(v.mean()) or 1.0
    return float(beta / scale * 100.0)


def _window_bool(ts: pd.Series, now: pd.Timestamp, minutes: int) -> pd.Series:
    cutoff = now - pd.Timedelta(minutes=minutes)
    return ts >= cutoff


def extract_features(
    history: pd.DataFrame,
    snapshot_by_id: dict[str, dict] | None = None,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Per-asset feature row.

    Columns:
      current_value, current_change_pct,
      change_5m, change_15m, change_1h, change_6h, change_24h,
      vol_1h, vol_24h, mean_1h, mean_24h, slope_1h,
      streak, n_obs_24h
    """
    if now is None:
        now = pd.Timestamp.utcnow().tz_localize(None) if history.empty else history["ts"].max()
    if hasattr(now, "tz_convert"):
        now = now.tz_convert("UTC")
    elif hasattr(now, "tz_localize") and now.tzinfo is None:
        now = now.tz_localize("UTC")

    rows: list[dict] = []
    for asset_id, g in history.groupby("asset_id", sort=False):
        g = g.sort_values("ts")
        vals = g["value"]
        ts = g["ts"]
        if vals.empty:
            continue

        now_val = float(vals.iloc[-1])
        row: dict = {"asset_id": asset_id, "current_value": now_val}

        for col_name, minutes in WINDOWS:
            mask = _window_bool(ts, now, minutes)
            window_vals = vals[mask]
            if len(window_vals) >= 1:
                row[col_name] = _pct_change(
                    now_val, float(window_vals.iloc[0])
                )
            else:
                row[col_name] = 0.0

        one_hour = vals[_window_bool(ts, now, 60)]
        one_day = vals[_window_bool(ts, now, 1440)]
        row["vol_1h"] = (
            float(one_hour.pct_change().std() * 100.0)
            if len(one_hour) >= 3
            else 0.0
        )
        row["vol_24h"] = (
            float(one_day.pct_change().std() * 100.0)
            if len(one_day) >= 3
            else 0.0
        )
        row["mean_1h"] = float(one_hour.mean()) if len(one_hour) else now_val
        row["mean_24h"] = float(one_day.mean()) if len(one_day) else now_val
        row["slope_1h"] = _slope(one_hour)

        chg = g["change_pct"].fillna(0.0)
        row["streak"] = _streak(chg.tail(30))
        row["n_obs_24h"] = int(len(one_day))
        rows.append(row)

    out = pd.DataFrame(rows).set_index("asset_id")

    # Fold in the live snapshot's current_change_pct (what the data-node
    # reports right now — possibly ahead of our last history row).
    if snapshot_by_id:
        live_change = {
            aid: float(s.get("changePct") or 0)
            for aid, s in snapshot_by_id.items()
        }
        out["current_change_pct"] = out.index.map(live_change).fillna(0.0)
    else:
        # Best-effort: tail change from history.
        out["current_change_pct"] = out.get("change_5m", 0.0)

    return out.fillna(0.0)
