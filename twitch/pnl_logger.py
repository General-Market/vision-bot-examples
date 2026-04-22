"""SQLite-backed PnL ledger. One row per batch join. Append-only.

Schema:
  ts           UNIX seconds when the join confirmed
  batch_id     on-chain batch id
  strategy     which predictor decided the picks
  deposit_wei  amount sent to parimutuel (18-dec USDC)
  bitmap_hash  keccak256 commitment
  tx_hash      join transaction hash
  picks_up     count of UP picks
  picks_total  total market count
  payout_wei   final payout (NULL until settlement observed)
  pnl_wei      payout - deposit (NULL until settled)
  settled_at   UNIX seconds of PlayerSettled event

Post-run reporting:
  python -c "from pnl_logger import report; report('pnl.db')"
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS joins (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           INTEGER NOT NULL,
    batch_id     INTEGER NOT NULL,
    strategy     TEXT NOT NULL,
    deposit_wei  TEXT NOT NULL,
    bitmap_hash  TEXT NOT NULL,
    tx_hash      TEXT NOT NULL,
    picks_up     INTEGER NOT NULL,
    picks_total  INTEGER NOT NULL,
    payout_wei   TEXT,
    pnl_wei      TEXT,
    settled_at   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_joins_unsettled ON joins(settled_at) WHERE settled_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_joins_batch ON joins(batch_id);
"""


class PnLLedger:
    def __init__(self, path: str | Path = "pnl.db") -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, isolation_level=None)
        self._conn.executescript(SCHEMA)

    def record_join(
        self,
        batch_id: int,
        strategy: str,
        deposit_wei: int,
        bitmap_hash: str,
        tx_hash: str,
        picks_up: int,
        picks_total: int,
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO joins(ts, batch_id, strategy, deposit_wei, bitmap_hash,
                                 tx_hash, picks_up, picks_total)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                int(time.time()),
                batch_id,
                strategy,
                str(deposit_wei),
                bitmap_hash,
                tx_hash,
                picks_up,
                picks_total,
            ),
        )
        return cur.lastrowid

    def record_settlement(self, join_id: int, payout_wei: int, deposit_wei: int) -> None:
        self._conn.execute(
            """UPDATE joins
                  SET payout_wei = ?, pnl_wei = ?, settled_at = ?
                WHERE id = ?""",
            (
                str(payout_wei),
                str(payout_wei - deposit_wei),
                int(time.time()),
                join_id,
            ),
        )

    def unsettled_since(self, since_ts: int = 0) -> list[dict]:
        rows = self._conn.execute(
            """SELECT id, batch_id, deposit_wei
                 FROM joins
                WHERE settled_at IS NULL AND ts >= ?""",
            (since_ts,),
        ).fetchall()
        return [
            {"id": r[0], "batch_id": r[1], "deposit_wei": int(r[2])} for r in rows
        ]


def report(db_path: str | Path = "pnl.db") -> None:
    con = sqlite3.connect(str(db_path))
    rows = con.execute(
        """SELECT strategy,
                  COUNT(*)                                  AS joins,
                  COUNT(settled_at)                         AS settled,
                  COALESCE(SUM(CAST(pnl_wei AS INTEGER)),0) AS pnl_sum_wei,
                  COALESCE(SUM(CAST(deposit_wei AS INTEGER)),0) AS wagered_wei
             FROM joins
            GROUP BY strategy"""
    ).fetchall()
    print(f"{'strategy':<14} {'joins':>7} {'settled':>8} {'pnl (USDC)':>14} {'wagered':>14} {'roi':>8}")
    for strategy, joins, settled, pnl_wei, wagered in rows:
        pnl = pnl_wei / 1e18
        wag = wagered / 1e18
        roi = (pnl / wag * 100) if wag else 0.0
        print(
            f"{strategy:<14} {joins:>7} {settled:>8} {pnl:>+14.4f} {wag:>14.4f} {roi:>+7.2f}%"
        )
