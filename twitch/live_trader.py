"""Live trading loop for the Twitch batch on Vision testnet.

Each iteration:
  1. Discover the current twitch batch and its config hash.
  2. Refresh the snapshot (always). Refresh history every FEATURE_REFRESH_SEC.
  3. Score every market with the configured predictor; binarise picks.
  4. Approve USDC if the allowance is insufficient.
  5. Submit joinBatchDirect, wait for receipt.
  6. Reveal bitmap to the oracle quorum.
  7. Log the join in SQLite.
  8. Sleep until next tick boundary.

A separate background pass reconciles unsettled joins against the
on-chain PlayerSettled event and stamps pnl_wei + settled_at.

Fails soft: any RPC / oracle hiccup logs a warning, waits, retries.
Never burns the whole loop because of a single bad tick.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from eth_account import Account
from web3 import Web3

from bitmap import encode_bitmap, hash_bitmap
from pnl_logger import PnLLedger
from strategy import FEATURE_STRATEGIES, make_predictor, make_predictor_with_features, picks_from_scores
from vision_bot import VisionBot, _retry_get


DEFAULTS = {
    "RPC_URL": "http://142.132.164.24/",
    "VISION_ADDRESS": "0x94d540bb45975bd5a0c7ba9a15a0d34e378f6c61",
    "DATA_NODE_URL": "https://generalmarket.io/bot-api",
    "ORACLE_URLS": "http://116.203.156.98/oracle1,http://116.203.156.98/oracle2,http://116.203.156.98/oracle3",
}


FEATURE_REFRESH_SEC = int(os.getenv("FEATURE_REFRESH_SEC", "900"))  # 15 min
IDLE_SLEEP_SEC = 10
TICK_DURATION = 60


def cfg(name: str) -> str:
    return os.getenv(name) or DEFAULTS[name]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


_stop = False


def _handle_sigint(signum, frame):
    global _stop
    _stop = True
    log("stop signal received — finishing current iteration")


signal.signal(signal.SIGINT, _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigint)


def ensure_allowance(bot: VisionBot, target_wei: int) -> None:
    """Approve a large allowance if the current one is below 10× deposit.
    ERC20 approvals are expensive — do it once per deploy, not per tick."""
    try:
        allowance = bot.usdc.functions.allowance(
            bot.bot_addr, bot.vision.address
        ).call()
    except Exception as e:
        log(f"  allowance check failed: {e} — approving defensively")
        allowance = 0
    if allowance < target_wei * 10:
        grant = 10**36
        log(f"  approving USDC {grant / 1e18:,.0f} → vision contract")
        bot.approve_usdc(grant)


_last_reconcile = 0.0


PORTFOLIO_API = "https://generalmarket.io/api/vision/player/{addr}/profile"
PORTFOLIO_URL = "https://generalmarket.io/profile/{addr}?tab=vision"


def verify_and_open_portfolio(
    bot_addr: str,
    auto_open: bool = True,
    retries: int = 4,
    backoff_sec: float = 3.0,
) -> str | None:
    """After the first successful join, poll the public profile API until
    the indexer has picked up at least one batch, then print (and
    optionally open) the wallet's portfolio URL. Returns the URL once
    trades are visible, or None if they don't appear within `retries`.
    """
    url = PORTFOLIO_URL.format(addr=bot_addr)
    api = PORTFOLIO_API.format(addr=bot_addr)
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(api, timeout=8)
            if r.ok:
                data = r.json() or {}
                total = int((data.get("stats") or {}).get("totalBatches", 0))
                if total >= 1:
                    log(
                        f"portfolio ready — {total} batch(es) indexed. "
                        f"View at: {url}"
                    )
                    if auto_open:
                        try:
                            import webbrowser
                            if webbrowser.open(url, new=2):
                                log("  opened in browser")
                        except Exception as e:
                            log(f"  (headless; open manually) {e}")
                    return url
        except Exception as e:
            log(f"portfolio poll attempt {attempt}: {e}")
        time.sleep(backoff_sec * attempt)
    log(
        f"portfolio not indexed yet — check later at: {url}"
    )
    return None


def reconcile_unsettled(bot: VisionBot, ledger: PnLLedger, lookback_blocks: int = 50_000, min_interval_sec: int = 120) -> None:
    """Walk unsettled joins, check for PlayerSettled, stamp pnl.
    Throttled — scanning logs on every tick is expensive and redundant."""
    global _last_reconcile
    if time.time() - _last_reconcile < min_interval_sec:
        return
    _last_reconcile = time.time()
    unsettled = ledger.unsettled_since(int(time.time()) - 7 * 86400)
    if not unsettled:
        return
    try:
        latest = bot.w3.eth.block_number
    except Exception:
        return
    from_block = max(0, latest - lookback_blocks)
    # One query per unique batch_id.
    by_batch: dict[int, list[dict]] = {}
    for u in unsettled:
        by_batch.setdefault(u["batch_id"], []).append(u)
    for batch_id, group in by_batch.items():
        try:
            logs = bot.vision.events.PlayerSettled.get_logs(
                argument_filters={"batchId": batch_id, "player": bot.bot_addr},
                from_block=from_block,
            )
        except Exception as e:
            log(f"  PlayerSettled scan failed for batch {batch_id}: {e}")
            continue
        for lg in logs:
            payout = int(lg["args"]["payout"])
            # Apply to the most recent unreconciled entry for this batch.
            if group:
                u = group.pop()
                ledger.record_settlement(u["id"], payout, u["deposit_wei"])
                pnl = (payout - u["deposit_wei"]) / 1e18
                log(f"  settled join#{u['id']} batch={batch_id} pnl={pnl:+.4f} USDC")


def run(args) -> int:
    load_dotenv(Path(__file__).parent / ".env")
    rpc = cfg("RPC_URL")
    vision_addr = cfg("VISION_ADDRESS")
    data_node = cfg("DATA_NODE_URL").rstrip("/")
    oracles = [u.strip().rstrip("/") for u in cfg("ORACLE_URLS").split(",") if u.strip()]

    key = os.getenv("BOT_PRIVATE_KEY")
    if not key:
        log("BOT_PRIVATE_KEY required to trade live. Exiting.")
        return 2
    bot = VisionBot(
        rpc_url=rpc, vision_address=vision_addr, private_key=key,
        data_node_url=data_node, oracles=oracles,
    )
    log(f"bot address: {bot.bot_addr}")

    # Balance check — warn if no USDC. With --auto-refaucet, top up from
    # the L3 public faucet whenever the balance falls below 2 × deposit
    # (so a long-running bot doesn't silently die mid-race).
    def _auto_top_up_if_needed() -> bool:
        try:
            bal_wei = bot.usdc.functions.balanceOf(bot.bot_addr).call()
        except Exception as e:
            log(f"balance check failed: {e}")
            return True
        if bal_wei >= int(args.deposit * 1e18):
            return True
        if args.auto_refaucet:
            log(f"balance {bal_wei/1e18:.4f} < deposit — auto-refaucet 1 USDC")
            try:
                faucet = Account.from_key(
                    "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"
                )
                nonce = bot.w3.eth.get_transaction_count(faucet.address)
                mint_abi = [{
                    "inputs": [{"name":"to","type":"address"},{"name":"amount","type":"uint256"}],
                    "name": "mint", "outputs": [],
                    "stateMutability": "nonpayable", "type": "function",
                }]
                u = bot.w3.eth.contract(address=bot.usdc.address, abi=mint_abi)
                tx = u.functions.mint(bot.bot_addr, int(1 * 1e18)).build_transaction({
                    "from": faucet.address, "nonce": nonce,
                    "gas": 120_000, "gasPrice": bot.w3.eth.gas_price,
                    "chainId": bot.w3.eth.chain_id,
                })
                signed = faucet.sign_transaction(tx)
                h = bot.w3.eth.send_raw_transaction(
                    getattr(signed, "raw_transaction", None) or signed.rawTransaction
                )
                bot.w3.eth.wait_for_transaction_receipt(h, timeout=60)
                log(f"  minted +1 USDC (tx {h.hex()[:18]}…)")
                return True
            except Exception as e:
                log(f"  auto-refaucet failed: {e}")
        log(f"insufficient balance ({bal_wei/1e18:.4f} USDC) — fund the wallet or pass --auto-refaucet")
        return False

    try:
        bal_wei = bot.usdc.functions.balanceOf(bot.bot_addr).call()
        log(f"L3 USDC balance: {bal_wei / 1e18:.4f}")
    except Exception as e:
        log(f"balance check failed: {e}")
    if not _auto_top_up_if_needed():
        return 3

    ledger = PnLLedger(args.db)
    deposit_wei = int(args.deposit * 1e18)
    ensure_allowance(bot, deposit_wei)

    features_df = None
    last_features_ts = 0
    markets_cache: dict[str, dict] = {}
    config_hash_cache: bytes | None = None

    last_joined_batch = ledger.last_joined_batch(args.strategy)
    if last_joined_batch >= 0:
        log(f"resuming — last joined batch was {last_joined_batch}")

    while not _stop:
        try:
            # Reconcile on each iteration — cheap when there's nothing.
            reconcile_unsettled(bot, ledger)
            if not _auto_top_up_if_needed():
                return 3

            source = bot.discover_source(args.source)
            config_hash = bytes.fromhex(source["configHash"][2:])
            tick = int(source.get("tickDurationSecs") or TICK_DURATION)
            markets = source.get("markets") or []
            markets_by_id = {m["assetId"]: m for m in markets if m.get("assetId")}
            n = len(markets)

            batch_id, on_chain_config = bot.find_active_batch_id(config_hash, tick_duration=tick)
            if batch_id == last_joined_batch:
                # Already bet this batch; wait for rotation.
                time.sleep(IDLE_SLEEP_SEC)
                continue

            snapshot_by_id = bot.fetch_snapshot(args.source)

            need_features = (
                args.strategy in FEATURE_STRATEGIES
                and (features_df is None
                     or time.time() - last_features_ts > FEATURE_REFRESH_SEC)
            )
            if need_features:
                from features import extract_features
                from history import fetch_history
                log(f"refreshing features: {n} assets × {args.history_hours}h history")
                hist = fetch_history(
                    data_node_url=data_node,
                    asset_ids=list(markets_by_id.keys()),
                    hours=args.history_hours,
                )
                features_df = extract_features(
                    hist, snapshot_by_id=snapshot_by_id, markets_by_id=markets_by_id,
                )
                last_features_ts = time.time()
                log(f"  features: {len(features_df)} assets")

            if args.strategy in FEATURE_STRATEGIES:
                predictor = make_predictor_with_features(
                    args.strategy,
                    features_df=features_df,
                    xgb_model_path=args.xgb_model,
                    markets_by_id=markets_by_id,
                )
            else:
                predictor = make_predictor(args.strategy)

            scores = predictor.predict(markets, snapshot_by_id)
            picks = picks_from_scores(scores, threshold=args.threshold)
            up = sum(1 for p in picks if p == "UP")
            log(f"batch={batch_id} picks UP={up}/{n} strategy={args.strategy}")

            bitmap = encode_bitmap(picks, n)
            bmhash = hash_bitmap(bitmap)

            log(f"  joining batch {batch_id} with deposit {args.deposit} USDC…")
            tx_hash = bot.join_batch(batch_id, on_chain_config, deposit_wei, bmhash)
            log(f"  join tx: {tx_hash.hex()}")

            accepted = 0
            try:
                accepted = bot.submit_bitmap(batch_id, bitmap, bmhash)
            except Exception as e:
                log(f"  oracle reveal failed: {e}")
            log(f"  oracle accepted: {accepted}/{len(oracles)}")

            ledger.record_join(
                batch_id=batch_id,
                strategy=args.strategy,
                deposit_wei=deposit_wei,
                bitmap_hash="0x" + bmhash.hex(),
                tx_hash=tx_hash.hex(),
                picks_up=up,
                picks_total=n,
            )
            last_joined_batch = batch_id
            join_count = (locals().get("join_count") or 0) + 1

            # First confirmed join — surface the portfolio URL so the
            # user can see their trades on generalmarket.io. Only opens
            # after the data-node indexer actually lists the batch.
            if join_count == 1:
                verify_and_open_portfolio(
                    bot.bot_addr,
                    auto_open=not args.no_open_portfolio,
                )

            if args.max_joins and join_count >= args.max_joins:
                log(f"reached --max-joins={args.max_joins}; exiting")
                return 0

        except Exception as e:
            log(f"iteration error: {e!r} — sleeping {IDLE_SLEEP_SEC}s")
            time.sleep(IDLE_SLEEP_SEC)
            continue

        # Wait until near the next tick boundary before looping.
        time.sleep(max(1, tick - 5))

    log("loop exited cleanly")
    return 0


def main():
    p = argparse.ArgumentParser(prog="live-trader")
    p.add_argument("--source", default="twitch")
    p.add_argument("--deposit", type=float, default=0.1)
    p.add_argument("--strategy", default="ensemble")
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--history-hours", type=int, default=6)
    p.add_argument("--xgb-model", default=os.getenv("XGB_MODEL_PATH", "models/xgb.pkl"))
    p.add_argument("--db", default="pnl.db")
    p.add_argument(
        "--auto-refaucet",
        action="store_true",
        help="Top up from the L3 testnet faucet whenever balance drops "
             "below deposit size. Testnet only.",
    )
    p.add_argument(
        "--max-joins",
        type=int,
        default=0,
        help="Exit cleanly after N successful joins (0 = run forever).",
    )
    p.add_argument(
        "--no-open-portfolio",
        action="store_true",
        help="Suppress auto-opening the portfolio URL after the first join. "
             "The URL is still printed to the log.",
    )
    args = p.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
