"""Print comparative PnL for two or more racing wallets.

Usage:
    python race_report.py pnl-ensemble.db pnl-all_yes.db [...]
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


def summarise(db: str) -> dict:
    empty = {"db": db, "strategy": "—", "joins": 0, "settled": 0,
             "pnl_usdc": 0.0, "wagered_usdc": 0.0, "roi_pct": 0.0,
             "span_hours": 0.0}
    if not Path(db).exists():
        return empty
    con = sqlite3.connect(db)
    try:
        row = con.execute(
            """SELECT COALESCE(strategy, '?'),
                      COUNT(*),
                      COUNT(settled_at),
                      COALESCE(SUM(CAST(pnl_wei     AS INTEGER)), 0),
                      COALESCE(SUM(CAST(deposit_wei AS INTEGER)), 0),
                      MIN(ts), MAX(COALESCE(settled_at, ts))
                 FROM joins"""
        ).fetchone()
    except sqlite3.OperationalError:
        return empty
    if not row or row[1] == 0:
        return empty
    strategy, joins, settled, pnl, wagered, t0, t1 = row
    pnl_u = pnl / 1e18
    wag_u = wagered / 1e18
    return {
        "db": db,
        "strategy": strategy,
        "joins": joins,
        "settled": settled,
        "pnl_usdc": pnl_u,
        "wagered_usdc": wag_u,
        "roi_pct": (pnl_u / wag_u * 100) if wag_u else 0.0,
        "span_hours": max(0.0, (t1 - t0) / 3600) if t0 and t1 else 0.0,
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    rows = [summarise(p) for p in sys.argv[1:]]
    fmt = "{strategy:<12} {joins:>7} {settled:>8} {pnl:>+13.4f} {wag:>13.4f} {roi:>+8.2f}% {span:>6.1f}h  {db}"
    print(f"{'strategy':<12} {'joins':>7} {'settled':>8} {'pnl USDC':>13} {'wagered':>13} {'ROI':>9} {'span':>7}  db")
    print("-" * 110)
    for r in rows:
        print(fmt.format(
            strategy=r["strategy"], joins=r["joins"], settled=r["settled"],
            pnl=r["pnl_usdc"], wag=r["wagered_usdc"], roi=r["roi_pct"],
            span=r["span_hours"], db=r["db"],
        ))
    # Head-to-head diff if exactly two wallets.
    if len(rows) == 2:
        diff = rows[0]["pnl_usdc"] - rows[1]["pnl_usdc"]
        tag = rows[0]["strategy"] + " vs " + rows[1]["strategy"]
        print("-" * 110)
        print(f"diff ({tag}): {diff:+.4f} USDC")


if __name__ == "__main__":
    main()
