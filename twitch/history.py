"""Bulk historical price fetcher for Vision sources.

Uses the data-node's `/market/batch-history` endpoint — up to 100 assets
per request, time-bounded. All 8192 twitch assets fit in ~82 chunked
requests with a concurrency bound.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Iterable

import pandas as pd
import requests

CHUNK = 100  # server-enforced cap


def _chunk(it: list[str], n: int):
    for i in range(0, len(it), n):
        yield it[i : i + n]


def _to_df(payload: dict) -> pd.DataFrame:
    data = payload.get("data") or {}
    rows = []
    for aid, entries in data.items():
        for e in entries:
            rows.append(
                {
                    "asset_id": aid,
                    "ts": pd.to_datetime(e.get("fetchedAt"), utc=True),
                    "value": float(e.get("value") or 0),
                    "change_pct": (
                        float(e["changePct"])
                        if e.get("changePct") is not None
                        else float("nan")
                    ),
                    "prev_close": (
                        float(e["prevClose"])
                        if e.get("prevClose") is not None
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def fetch_history(
    data_node_url: str,
    asset_ids: Iterable[str],
    hours: int = 24,
    max_workers: int = 8,
    timeout: int | None = None,
) -> pd.DataFrame:
    # Scale timeout with window size — 1 week fetches can push 700k rows/chunk.
    if timeout is None:
        timeout = 30 if hours <= 24 else (60 if hours <= 72 else 180)
    """Pull the last `hours` of price history for every asset, in chunks.

    Returns a DataFrame with columns: asset_id, ts, value, change_pct, prev_close.
    Sorted by (asset_id, ts). Empty if the endpoint refuses every chunk.
    """
    assets = [a for a in asset_ids if a]
    now = datetime.now(timezone.utc)
    frm = (now - timedelta(hours=hours)).isoformat().replace("+00:00", "Z")
    base = data_node_url.rstrip("/")

    def fetch_one(chunk: list[str]) -> pd.DataFrame:
        params = {"assets": ",".join(chunk), "from": frm}
        try:
            r = requests.get(
                f"{base}/market/batch-history",
                params=params,
                timeout=timeout,
            )
            if r.status_code != 200:
                return pd.DataFrame()
            return _to_df(r.json())
        except requests.RequestException:
            return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    chunks = list(_chunk(assets, CHUNK))
    if not chunks:
        return pd.DataFrame(
            columns=["asset_id", "ts", "value", "change_pct", "prev_close"]
        )

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_one, c): c for c in chunks}
        done = 0
        for fut in as_completed(futures):
            df = fut.result()
            if not df.empty:
                frames.append(df)
            done += 1
            if done % 10 == 0:
                print(
                    f"  history fetch: {done}/{len(chunks)} chunks "
                    f"({time.time()-t0:.1f}s)"
                )

    if not frames:
        return pd.DataFrame(
            columns=["asset_id", "ts", "value", "change_pct", "prev_close"]
        )
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["asset_id", "ts"]).sort_values(
        ["asset_id", "ts"]
    )
    return out
