"""Polymarket CLOB telemetry — orderbook snapshots and recent trades.

Replaces twitch's `viewer_telemetry`. Same role: pull live, low-cost
signals from a public endpoint that the daily Gamma loader doesn't
expose at the resolution we need.
"""
from __future__ import annotations

import logging
import time

import pandas as pd
import requests

from config.settings import config

log = logging.getLogger(__name__)


class MarketTelemetry:
    """Per-market orderbook + recent-trade snapshots from the CLOB."""

    def __init__(self, clob_url: str | None = None):
        self.base = (clob_url or config.polymarket.clob_url).rstrip("/")

    def fetch_orderbook(self, token_id: str) -> dict:
        """Return midprice, depth, and spread for a single token id.

        Polymarket binary markets have two token ids — one for YES, one
        for NO. This call expects the YES token. Spread and depth are
        derived from the top-of-book.
        """
        try:
            r = requests.get(
                f"{self.base}/book", params={"token_id": token_id}, timeout=10
            )
            r.raise_for_status()
            book = r.json() or {}
        except requests.RequestException as e:
            log.warning("CLOB book fetch failed %s: %s", token_id, e)
            time.sleep(0.2)
            return {}

        bids = book.get("bids", []) or []
        asks = book.get("asks", []) or []
        best_bid = float(bids[0]["price"]) if bids else 0.0
        best_ask = float(asks[0]["price"]) if asks else 1.0
        mid = (best_bid + best_ask) / 2.0 if (bids and asks) else 0.5
        spread = max(0.0, best_ask - best_bid)
        bid_depth = sum(float(b.get("size", 0)) for b in bids[:5])
        ask_depth = sum(float(a.get("size", 0)) for a in asks[:5])
        time.sleep(0.05)

        return {
            "TOKEN_ID": token_id,
            "MID_PRICE": mid,
            "SPREAD": spread,
            "BID_DEPTH_5": bid_depth,
            "ASK_DEPTH_5": ask_depth,
            "DEPTH_IMBALANCE": (bid_depth - ask_depth) / max(bid_depth + ask_depth, 1.0),
        }

    def fetch_recent_trades(
        self, token_id: str, limit: int = 100
    ) -> pd.DataFrame:
        cols = ["TS", "PRICE", "SIZE", "SIDE"]
        try:
            r = requests.get(
                f"{self.base}/trades",
                params={"market": token_id, "limit": limit},
                timeout=15,
            )
            r.raise_for_status()
            data = r.json() or []
        except requests.RequestException as e:
            log.warning("CLOB trades fetch failed %s: %s", token_id, e)
            time.sleep(0.2)
            return pd.DataFrame(columns=cols)

        rows = []
        for t in data if isinstance(data, list) else data.get("trades", []):
            rows.append({
                "TS": pd.to_datetime(t.get("timestamp"), unit="s", errors="coerce"),
                "PRICE": float(t.get("price", 0) or 0),
                "SIZE": float(t.get("size", 0) or 0),
                "SIDE": t.get("side", ""),
            })

        time.sleep(0.05)
        return pd.DataFrame(rows, columns=cols)


# Backwards-compat alias for any code paths still importing the twitch name.
ViewerTelemetry = MarketTelemetry
