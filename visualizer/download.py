#!/usr/bin/env python3
"""
Download Vision bot trade data + underlying price history into a flat
directory the static HTML viewer can consume.

Reads:
  - pnl.json (active + history positions, with bets per market)
  - oracle /vision/batches  (to map batch_id -> config_hash, source_id)
  - data-node /batches/config/{config_hash}  (ordered asset list)
  - data-node /vision/batch/{id}/history?days=N  (price series per asset)
  - oracle /vision/balance/{id}/{player}  (final balance, settled flag)

Writes:
  - <out>/index.json                       (one row per traded asset, lite)
  - <out>/data/<batch_id>/<asset_id>.json  (price series + trade meta)

Usage:
    python download.py [--pnl pnl.json] [--config config.toml] [--out . ]
                       [--days 7] [--player 0xabc...] [--max-batches 200]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("viz")

DEFAULTS = {
    # Public General Market endpoints. The frontend at generalmarket.io
    # proxies the data-node and oracle surfaces under /api/. Override with
    # --data-node / --oracle if you run your own node locally.
    "data_node": "https://generalmarket.io/api",
    "oracle_urls": ["https://generalmarket.io/api"],
    "rpc_url": "https://rpc.generalmarket.io/",
    "pnl_file": "pnl.json",
}

SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_name(s: str) -> str:
    return SAFE_NAME.sub("_", s)[:120] or "_"


# ── Config ─────────────────────────────────────────────────────────

def load_config(path: Optional[str]) -> dict:
    cfg = dict(DEFAULTS)
    if path and os.path.exists(path):
        try:
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib  # type: ignore
            with open(path, "rb") as f:
                raw = tomllib.load(f)
            for k in ("data_node", "rpc_url", "pnl_file"):
                if k in raw:
                    cfg[k] = raw[k]
            if isinstance(raw.get("oracle_urls"), list) and raw["oracle_urls"]:
                cfg["oracle_urls"] = raw["oracle_urls"]
        except Exception as e:
            log.warning("Config %s unreadable (%s) — using defaults", path, e)
    return cfg


def load_pnl(path: str) -> dict:
    if not os.path.exists(path):
        log.error("PnL file not found: %s", path)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


# ── Oracle / data-node helpers ─────────────────────────────────────

def oracle_get(urls: list[str], path: str, timeout: float = 10) -> Optional[dict]:
    for url in urls:
        try:
            resp = requests.get(f"{url.rstrip('/')}{path}", timeout=timeout)
            if resp.ok:
                return resp.json()
        except requests.RequestException:
            continue
    return None


def fetch_batches(oracle_urls: list[str]) -> dict[int, dict]:
    """Map batch_id -> {config_hash, source_id, market_count}."""
    data = oracle_get(oracle_urls, "/vision/batches")
    if not data:
        return {}
    out = {}
    for b in data.get("batches", []):
        bid = b.get("id")
        if bid is None:
            continue
        out[int(bid)] = {
            "config_hash": b.get("config_hash") or b.get("configHash") or "",
            "source_id": b.get("source_id") or b.get("sourceId") or "",
            "market_count": b.get("market_count", 0),
            "tick_duration": b.get("tick_duration") or b.get("tickDuration"),
        }
    return out


def fetch_config(data_node: str, config_hash: str) -> Optional[dict]:
    if not config_hash:
        return None
    if not config_hash.startswith("0x"):
        config_hash = "0x" + config_hash
    try:
        resp = requests.get(f"{data_node}/batches/config/{config_hash}", timeout=15)
        if resp.ok:
            return resp.json()
    except requests.RequestException:
        return None
    return None


def fetch_history(data_node: str, batch_id: int, days: int) -> Optional[dict]:
    try:
        resp = requests.get(
            f"{data_node}/vision/batch/{batch_id}/history",
            params={"days": days},
            timeout=60,
        )
        if resp.ok:
            return resp.json()
    except requests.RequestException as e:
        log.warning("Batch %d: history fetch failed: %s", batch_id, e)
    return None


def fetch_balance(oracle_urls: list[str], batch_id: int, player: str) -> Optional[dict]:
    return oracle_get(oracle_urls, f"/vision/balance/{batch_id}/{player}")


# ── Position parsing ───────────────────────────────────────────────

def collect_positions(pnl: dict) -> list[dict]:
    """Flatten active + history into a single list of position dicts."""
    out = []
    for p in pnl.get("active", []) or []:
        if isinstance(p, dict):
            out.append({**p, "_source": "active"})
    for p in pnl.get("history", []) or []:
        if isinstance(p, dict):
            out.append({**p, "_source": "history"})
    return out


# ── Main pipeline ──────────────────────────────────────────────────

def build(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    pnl_path = args.pnl or cfg["pnl_file"]
    pnl = load_pnl(pnl_path)
    positions = collect_positions(pnl)
    if not positions:
        log.error("No positions in %s — nothing to visualize", pnl_path)
        sys.exit(1)

    data_node = (args.data_node or cfg["data_node"]).rstrip("/")
    oracle_urls = args.oracle or cfg["oracle_urls"]

    # batch_id -> config_hash, source_id (best-effort: the oracle may have
    # forgotten old paused batches). Positions also carry no config_hash, so
    # if the oracle dropped a batch the asset names degrade to ids only.
    log.info("Fetching active batch directory from %d oracle(s)", len(oracle_urls))
    chain_batches = fetch_batches(oracle_urls)
    log.info("  %d active batches in oracle directory", len(chain_batches))

    # Optional fallback: a JSON file mapping batch_id -> {config_hash, source_id,
    # tick_duration}. Useful when the oracle's live /vision/batches has dropped
    # settled batches but we still want to visualize them.
    if args.batches_meta and os.path.exists(args.batches_meta):
        try:
            with open(args.batches_meta) as f:
                meta_doc = json.load(f)
            merged = 0
            for k, v in meta_doc.items():
                bid = int(k)
                if bid not in chain_batches and isinstance(v, dict):
                    chain_batches[bid] = {
                        "config_hash": v.get("config_hash", ""),
                        "source_id": v.get("source_id", ""),
                        "tick_duration": v.get("tick_duration"),
                        "market_count": v.get("market_count", 0),
                    }
                    merged += 1
            log.info("  merged %d batches from %s", merged, args.batches_meta)
        except Exception as e:
            log.warning("Failed to read batches-meta %s: %s", args.batches_meta, e)

    # Cap batches to avoid pulling thousands of histories on a noisy bot.
    batch_ids = sorted({int(p["batch_id"]) for p in positions if "batch_id" in p})
    if args.max_batches and len(batch_ids) > args.max_batches:
        batch_ids = batch_ids[-args.max_batches:]
        log.info("Capping to %d most recent batches", len(batch_ids))

    out_root = os.path.abspath(args.out)
    data_root = os.path.join(out_root, "data")
    os.makedirs(data_root, exist_ok=True)

    # ── Resolve config + history per batch in parallel ─────────────
    log.info("Resolving config + history for %d batches", len(batch_ids))

    def resolve(bid: int):
        meta = chain_batches.get(bid, {})
        config_hash = meta.get("config_hash", "")
        cfg_doc = fetch_config(data_node, config_hash) if config_hash else None
        hist = fetch_history(data_node, bid, args.days)
        bal = None
        if args.player:
            bal = fetch_balance(oracle_urls, bid, args.player)
        return bid, meta, cfg_doc, hist, bal

    by_batch = {}
    with ThreadPoolExecutor(max_workers=min(16, max(2, len(batch_ids)))) as ex:
        futures = {ex.submit(resolve, bid): bid for bid in batch_ids}
        for i, fut in enumerate(as_completed(futures)):
            bid, meta, cfg_doc, hist, bal = fut.result()
            by_batch[bid] = {"meta": meta, "config": cfg_doc, "history": hist, "balance": bal}
            if (i + 1) % 25 == 0 or (i + 1) == len(futures):
                log.info("  %d/%d batches resolved", i + 1, len(futures))

    # ── Build per-asset JSON files + index ─────────────────────────
    index_items: list[dict] = []
    written = 0

    pos_by_batch: dict[int, dict] = {}
    # Prefer the most recent record per batch (active wins over history if both).
    for p in positions:
        bid = int(p.get("batch_id", -1))
        if bid < 0:
            continue
        if bid not in pos_by_batch or p.get("_source") == "active":
            pos_by_batch[bid] = p

    for bid, info in by_batch.items():
        cfg_doc = info["config"] or {}
        hist = info["history"] or {}
        bal = info["balance"] or {}
        pos = pos_by_batch.get(bid, {})

        # Ordered asset list (canonical bit order)
        markets_cfg = cfg_doc.get("markets") or []
        asset_ids: list[str] = []
        names: dict[str, str] = {}
        for m in markets_cfg:
            if isinstance(m, dict) and m.get("assetId"):
                aid = m["assetId"]
                asset_ids.append(aid)
                names[aid] = m.get("name") or m.get("symbol") or aid

        # Price series per asset
        series_by_id: dict[str, list[dict]] = {}
        for m in hist.get("markets", []) or []:
            aid = m.get("id") or m.get("assetId")
            if not aid:
                continue
            cleaned = []
            for p in m.get("prices") or []:
                try:
                    ts = int(p["ts"])
                    price = float(p["price"])
                except (KeyError, TypeError, ValueError):
                    continue
                cleaned.append({"ts": ts, "price": price})
            cleaned.sort(key=lambda x: x["ts"])
            series_by_id[aid] = cleaned

        bets = pos.get("bets") or []
        joined_at = pos.get("joined_at")
        deposit_wei = int(pos.get("deposited", 0) or 0)
        balance_wei = int(bal.get("balance", pos.get("balance", deposit_wei)) or 0)
        settled = bool(bal.get("settled", pos.get("_source") == "history"))
        usdc_dec = 18
        deposit_usdc = deposit_wei / 10 ** usdc_dec if deposit_wei else None
        balance_usdc = balance_wei / 10 ** usdc_dec if balance_wei else None
        pnl_usdc = (balance_usdc - deposit_usdc) if (deposit_usdc is not None and balance_usdc is not None) else None

        source_id = info["meta"].get("source_id", "")
        tick_duration = info["meta"].get("tick_duration")

        # Use the canonical bit order from config; if config is missing,
        # fall back to whatever order the history endpoint returned.
        order = asset_ids or list(series_by_id.keys())

        for idx, aid in enumerate(order):
            series = series_by_id.get(aid) or []
            bet = bets[idx] if idx < len(bets) else None
            asset_name = names.get(aid, aid)

            # Skip empties: no price history AND no trade — pointless to list.
            if not series and bet is None:
                continue

            asset_doc = {
                "batch_id": bid,
                "asset_id": aid,
                "asset_name": asset_name,
                "source_id": source_id,
                "tick_duration": tick_duration,
                "history": series,
                "trade": {
                    "joined_at": joined_at,
                    "bet": bet,
                    "deposit_usdc": deposit_usdc,
                    "balance_usdc": balance_usdc,
                    "pnl_usdc": pnl_usdc,
                    "settled": settled,
                    "market_count": len(order),
                },
            }

            batch_dir = os.path.join(data_root, str(bid))
            os.makedirs(batch_dir, exist_ok=True)
            file_path = os.path.join(batch_dir, f"{safe_name(aid)}.json")
            with open(file_path, "w") as f:
                json.dump(asset_doc, f, separators=(",", ":"))
            written += 1

            # Index row — small, quick to scan, no heavy series.
            index_items.append({
                "batch_id": bid,
                "asset_id": aid,
                "asset_name": asset_name,
                "source_id": source_id,
                "bet": bet,
                "pnl_usdc": pnl_usdc,
                "settled": settled,
                "points": len(series),
                "joined_at": joined_at,
                "file": f"data/{bid}/{safe_name(aid)}.json",
            })

    # Stable, predictable order: most recent batch first, then asset name.
    index_items.sort(
        key=lambda r: (-(r.get("joined_at") or 0), r.get("asset_name") or ""),
    )

    index_doc = {
        "generated_at": int(time.time()),
        "usdc_decimals": 18,
        "batch_count": len(by_batch),
        "asset_count": len(index_items),
        "items": index_items,
    }
    with open(os.path.join(out_root, "index.json"), "w") as f:
        json.dump(index_doc, f, separators=(",", ":"))

    log.info(
        "Done. %d batches, %d asset files, written to %s",
        len(by_batch), written, out_root,
    )
    log.info("Reload the Vite app — data in %s", out_root)


def main() -> None:
    ap = argparse.ArgumentParser(description="Vision bot data + history downloader")
    ap.add_argument("--pnl", help="Path to pnl.json (default: from config or pnl.json)")
    ap.add_argument("--config", default="../config.toml", help="config.toml path (default: ../config.toml)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "public"),
                    help="Output directory (default: ./public — served by Vite)")
    ap.add_argument("--data-node", help="Override data-node URL")
    ap.add_argument("--oracle", action="append", help="Oracle URL (repeatable)")
    ap.add_argument("--days", type=int, default=7, help="History window in days (default: 7)")
    ap.add_argument("--player", help="Player address — fetches final balance per batch")
    ap.add_argument("--max-batches", type=int, default=500,
                    help="Hard cap on batches resolved (default: 500)")
    ap.add_argument("--batches-meta", help="JSON file: {batch_id: {config_hash, source_id, tick_duration}}. "
                    "Used as fallback when oracle dropped settled batches.")
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
