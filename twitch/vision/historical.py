"""Historical Vision pool data from the data-node API."""
from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from config.settings import config


class VisionHistorical:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or config.data_node_url).rstrip("/")

    def get_price_history(
        self,
        source: str,
        batch_id: int,
        lookback_hours: int = 48,
    ) -> pd.DataFrame:
        url = f"{self.base_url}/api/vision/batches/{source}/history"
        params = {"batch_id": batch_id, "hours": lookback_hours}
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            payload = r.json()
        except requests.RequestException as e:
            print(f"[vision-hist] history fetch failed: {e}")
            return pd.DataFrame(
                columns=["timestamp", "yes_price", "no_price", "liquidity_usdc"]
            )

        rows: list[dict[str, Any]] = payload.get("history", []) or []
        if not rows:
            return pd.DataFrame(
                columns=["timestamp", "yes_price", "no_price", "liquidity_usdc"]
            )

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["t"], unit="s", utc=True)
        df["yes_price"] = df["yes"].astype(float)
        df["no_price"] = df["no"].astype(float)
        df["liquidity_usdc"] = df["liq"].astype(float) / 1e18
        return df[["timestamp", "yes_price", "no_price", "liquidity_usdc"]]

    def get_pool_depth(self, batch_id: int) -> dict[str, Any]:
        url = f"{self.base_url}/api/vision/batches/{batch_id}/depth"
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            payload = r.json()
        except requests.RequestException as e:
            print(f"[vision-hist] depth fetch failed: {e}")
            return {
                "yes_pool_usdc": 0.0,
                "no_pool_usdc": 0.0,
                "total_pool_usdc": 0.0,
                "imbalance": 0.0,
                "spread_pct": 0.0,
            }

        yes_pool = float(payload.get("yes_pool", 0)) / 1e18
        no_pool = float(payload.get("no_pool", 0)) / 1e18
        total = yes_pool + no_pool
        imbalance = (yes_pool - no_pool) / total if total > 0 else 0.0
        return {
            "yes_pool_usdc": yes_pool,
            "no_pool_usdc": no_pool,
            "total_pool_usdc": total,
            "imbalance": imbalance,
            "spread_pct": 0.0,
        }
