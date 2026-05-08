"""Short-horizon feature extraction for Vision markets.

Tick duration is 60 s, so predictive signal lives in short windows
(1m / 5m / 15m). Longer horizons (1h+) are retained only where the
resolution rule requires them — the 24h baseline is the `thresholdSource`
the oracle uses to resolve up_x / down_x markets.

Features emitted per asset_id:
  change_1m, change_5m, change_15m        short-window pct changes
  vol_5m                                  std of change_pct in last 5m
  slope_5m                                normalised LR slope over 5m
  streak                                  current same-sign run length
  n_obs_5m                                sample density in last 5m
  hour_utc, day_of_week, is_weekend,      temporal context
  is_primetime                            UTC 18–23 = primetime
  current_value, baseline_24h             level + resolution baseline
  dist_to_up, dist_to_down                signed distance to boundary
  current_resolution                      would THIS tick resolve YES right now
  category_mean_5m, asset_vs_category_5m  cross-asset platform signal
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SHORT_WINDOWS = [
    ("change_1m", 1),
    ("change_5m", 5),
    ("change_15m", 15),
]

# Baseline window matches `thresholdSource: "24h_history"` — used only to
# compute dist_to_threshold and current_resolution. Not a change feature.
BASELINE_MINUTES = 1440

PRIMETIME_START_UTC = 18
PRIMETIME_END_UTC = 23


def _pct_change(now: float, then: float) -> float:
    if then == 0 or not np.isfinite(then) or then is None:
        return 0.0
    return (now - then) / then * 100.0


def _streak(changes: pd.Series) -> int:
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
    if len(values) < 3:
        return 0.0
    v = values.values.astype(float)
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


def _resolution_at(
    value: float,
    baseline: float,
    res_type: str | None,
    bps: float | None,
) -> int:
    """Would this market resolve YES right now? 1 = YES, 0 = NO."""
    if not res_type or bps is None:
        return 0
    frac = float(bps) / 10000.0
    if res_type == "up_x":
        return 1 if value > baseline * (1 + frac) else 0
    if res_type == "down_x":
        return 1 if baseline > 0 and value < baseline * (1 - frac) else 0
    if res_type == "up_0":
        return 1 if value > 0 else 0
    if res_type == "down_0":
        return 1 if value < 0 else 0
    if res_type == "flat_x":
        if baseline == 0 or not np.isfinite(baseline):
            return 0
        drift = abs(value - baseline) / abs(baseline)
        return 1 if drift < frac else 0
    return 0


def compute_label(
    next_value: float,
    baseline: float,
    res_type: str | None,
    bps: float | None,
) -> int:
    """Ground-truth resolution label for training — same rule the oracle
    uses at settlement."""
    return _resolution_at(next_value, baseline, res_type, bps)


def _baseline_24h(ts: pd.Series, vals: pd.Series, now: pd.Timestamp) -> float:
    """Value closest to `now - 24h`. Oracle uses 24h_history as default."""
    target = now - pd.Timedelta(minutes=BASELINE_MINUTES)
    mask = ts <= target
    if mask.any():
        return float(vals[mask].iloc[-1])
    return float(vals.iloc[0])


def _window_bool(ts: pd.Series, now: pd.Timestamp, minutes: int) -> pd.Series:
    cutoff = now - pd.Timedelta(minutes=minutes)
    return ts >= cutoff


def extract_features(
    history: pd.DataFrame,
    snapshot_by_id: dict[str, dict] | None = None,
    markets_by_id: dict[str, dict] | None = None,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if now is None:
        now = history["ts"].max() if not history.empty else pd.Timestamp.utcnow()
    if hasattr(now, "tz_convert"):
        now = now.tz_convert("UTC")
    elif getattr(now, "tzinfo", None) is None:
        now = pd.Timestamp(now).tz_localize("UTC")

    rows: list[dict] = []
    for asset_id, g in history.groupby("asset_id", sort=False):
        g = g.sort_values("ts")
        vals = g["value"]
        ts = g["ts"]
        if vals.empty:
            continue

        now_val = float(vals.iloc[-1])
        row: dict = {"asset_id": asset_id, "current_value": now_val}

        for col_name, minutes in SHORT_WINDOWS:
            window_vals = vals[_window_bool(ts, now, minutes)]
            if len(window_vals) >= 1:
                row[col_name] = _pct_change(now_val, float(window_vals.iloc[0]))
            else:
                row[col_name] = 0.0

        five_min = vals[_window_bool(ts, now, 5)]
        chg_5m = g["change_pct"][_window_bool(ts, now, 5)]
        row["vol_5m"] = float(chg_5m.std()) if len(chg_5m) >= 3 else 0.0
        row["slope_5m"] = _slope(five_min)
        row["n_obs_5m"] = int(len(five_min))

        row["streak"] = _streak(g["change_pct"].fillna(0.0).tail(30))

        row["hour_utc"] = int(now.hour)
        row["day_of_week"] = int(now.dayofweek)
        row["is_weekend"] = int(now.dayofweek >= 5)
        row["is_primetime"] = int(
            PRIMETIME_START_UTC <= now.hour <= PRIMETIME_END_UTC
        )

        baseline = _baseline_24h(ts, vals, now)
        row["baseline_24h"] = baseline

        market = (markets_by_id or {}).get(asset_id) or {}
        res_type = market.get("resolutionType")
        bps = market.get("thresholdBps")

        if res_type == "up_x" and bps is not None and baseline > 0:
            target = baseline * (1 + float(bps) / 10000.0)
            row["dist_to_up"] = (now_val - target) / (abs(target) or 1.0) * 100.0
            row["dist_to_down"] = 0.0
        elif res_type == "down_x" and bps is not None and baseline > 0:
            target = baseline * (1 - float(bps) / 10000.0)
            row["dist_to_down"] = (target - now_val) / (abs(target) or 1.0) * 100.0
            row["dist_to_up"] = 0.0
        else:
            row["dist_to_up"] = 0.0
            row["dist_to_down"] = 0.0

        row["current_resolution"] = _resolution_at(
            now_val, baseline, res_type, bps
        )

        rows.append(row)

    out = pd.DataFrame(rows).set_index("asset_id")
    if out.empty:
        return out

    if snapshot_by_id:
        live_change = {
            aid: float(s.get("changePct") or 0)
            for aid, s in snapshot_by_id.items()
        }
        out["current_change_pct"] = out.index.map(live_change).fillna(0.0)
        cats = {
            aid: (s.get("category") or "unknown")
            for aid, s in snapshot_by_id.items()
        }
        out["_category"] = out.index.map(cats).fillna("unknown")
    else:
        out["current_change_pct"] = out.get("change_5m", 0.0)
        out["_category"] = "unknown"

    cat_mean = out.groupby("_category")["change_5m"].transform("mean")
    out["category_mean_5m"] = cat_mean.fillna(0.0)
    out["asset_vs_category_5m"] = out["change_5m"] - out["category_mean_5m"]
    out = out.drop(columns=["_category"])

    return out.fillna(0.0)
