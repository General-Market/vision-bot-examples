"""Polymarket Gamma loader: paginates closed events by category."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from config.settings import config

log = logging.getLogger(__name__)


class PolymarketDataLoader:
    """Pulls historical settled events from Polymarket Gamma.

    A Polymarket *event* groups one or more *markets*. For binary YES/NO
    events the event resolves with `closedTime` and the winning outcome
    can be inferred from `outcomePrices` (last printed price near 1.0).

    The loader returns one row per market, keyed by `MARKET_ID`, with
    enough features to mirror the twitch loader's output shape:
      - timestamp (CLOSED_TIME) for cleaner.sort
      - a numeric outcome column (FINAL_YES_PRICE) for label_outcome
      - metadata (CATEGORY, VOLUME, LIQUIDITY) for downstream features
    """

    def __init__(
        self,
        gamma_url: str | None = None,
        categories: list[str] | None = None,
    ):
        self.gamma_url = (gamma_url or config.polymarket.gamma_url).rstrip("/")
        self.categories = categories or config.polymarket.categories
        self.headers = {"Accept": "application/json"}

    def _get(self, path: str, params: dict) -> list[dict] | None:
        try:
            r = requests.get(
                f"{self.gamma_url}{path}",
                headers=self.headers,
                params=params,
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()
            return data if isinstance(data, list) else data.get("data", [])
        except requests.RequestException as e:
            log.warning("Polymarket GET failed %s: %s", path, e)
            return None

    @staticmethod
    def _outcome(market: dict) -> float | None:
        """Final YES price for a settled binary market.

        Polymarket markets carry `outcomes` (e.g. ["Yes","No"]) and
        `outcomePrices` (e.g. ["1","0"]). When closed, the YES outcome
        sits at 1.0 if YES won, 0.0 otherwise.
        """
        prices = market.get("outcomePrices") or []
        if isinstance(prices, str):
            try:
                import json as _json
                prices = _json.loads(prices)
            except Exception:
                return None
        if not prices:
            return None
        try:
            return float(prices[0])
        except (TypeError, ValueError):
            return None

    def load_category_events(
        self, category: str, lookback_days: int = 120
    ) -> pd.DataFrame:
        cols = [
            "MARKET_ID", "EVENT_ID", "QUESTION", "SLUG",
            "CATEGORY", "CREATED_AT", "END_DATE", "CLOSED_TIME",
            "VOLUME", "LIQUIDITY", "FINAL_YES_PRICE", "OUTCOME",
        ]
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        rows: list[dict] = []
        offset = 0
        page_size = 100
        max_events = config.polymarket.max_events_per_category

        while len(rows) < max_events:
            params = {
                "closed": "true",
                "tag_slug": category,
                "limit": page_size,
                "offset": offset,
                "ascending": "false",
                "order": "endDate",
            }
            page = self._get("/markets", params)
            time.sleep(0.1)
            if not page:
                break

            page_done = False
            for m in page:
                end_iso = m.get("endDate") or m.get("closedTime") or ""
                try:
                    end_dt = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
                    end_naive = end_dt.replace(tzinfo=None)
                except (ValueError, TypeError):
                    continue
                if end_naive < cutoff:
                    page_done = True
                    break

                yes = self._outcome(m)
                if yes is None:
                    continue

                created_iso = m.get("createdAt") or m.get("startDate") or ""
                try:
                    created_dt = datetime.fromisoformat(
                        created_iso.replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                except (ValueError, TypeError):
                    created_dt = end_naive

                rows.append({
                    "MARKET_ID": m.get("conditionId") or m.get("id"),
                    "EVENT_ID": m.get("eventId") or m.get("event_id"),
                    "QUESTION": m.get("question", ""),
                    "SLUG": m.get("slug", ""),
                    "CATEGORY": category,
                    "CREATED_AT": created_dt,
                    "END_DATE": end_naive,
                    "CLOSED_TIME": end_naive,
                    "VOLUME": float(m.get("volume", 0) or 0),
                    "LIQUIDITY": float(m.get("liquidity", 0) or 0),
                    "FINAL_YES_PRICE": yes,
                    "OUTCOME": int(yes >= 0.5),
                })

            if page_done or len(page) < page_size:
                break
            offset += page_size

        return pd.DataFrame(rows, columns=cols)

    def load_all(self, lookback_days: int = 120) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for cat in self.categories:
            df = self.load_category_events(cat, lookback_days=lookback_days)
            print(f"[polymarket] {cat}: {len(df)} settled markets")
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
