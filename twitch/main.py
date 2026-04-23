import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from eth_account import Account
from web3 import Web3

from bitmap import encode_bitmap, hash_bitmap
from strategy import (
    ALL_STRATEGIES,
    FEATURE_STRATEGIES,
    REGISTRY,
    make_predictor,
    make_predictor_with_features,
    pick_summary,
    picks_from_scores,
)
from vision_bot import VisionBot, _retry_get

load_dotenv(Path(__file__).parent / ".env")

DEFAULTS = {
    "RPC_URL": "http://142.132.164.24/",
    "VISION_ADDRESS": "0x94d540bb45975bd5a0c7ba9a15a0d34e378f6c61",
    "DATA_NODE_URL": "https://generalmarket.io/bot-api",
    "ORACLE_URLS": "http://116.203.156.98/oracle1,http://116.203.156.98/oracle2,http://116.203.156.98/oracle3",
}


def cfg(name: str) -> str:
    return os.getenv(name) or DEFAULTS[name]


def get_key(subcommand: str) -> str:
    key = os.getenv("BOT_PRIVATE_KEY")
    if key:
        return key
    if subcommand in ("probe", "dryrun"):
        acct = Account.create()
        print(f"[ephemeral key] address={acct.address}")
        return acct.key.hex()
    print("BOT_PRIVATE_KEY is required for 'trade'.")
    sys.exit(2)


def fail_http(url: str, resp) -> None:
    print(f"HTTP {resp.status_code} from {url}")
    print(resp.text[:600])
    sys.exit(2)


def cmd_probe(args):
    rpc = cfg("RPC_URL")
    vision_addr = cfg("VISION_ADDRESS")
    data_node = cfg("DATA_NODE_URL").rstrip("/")
    oracles = [u.strip().rstrip("/") for u in cfg("ORACLE_URLS").split(",") if u.strip()]

    w3 = Web3(Web3.HTTPProvider(rpc))
    if not w3.is_connected():
        print(f"RPC unreachable: {rpc}")
        sys.exit(2)

    chain_id = w3.eth.chain_id
    print(f"chain_id: {chain_id}")

    code = w3.eth.get_code(Web3.to_checksum_address(vision_addr))
    print(f"vision bytecode length: {len(code)}")
    if len(code) == 0:
        print("Vision contract has no bytecode. Address is dead or wrong.")
        sys.exit(2)

    key = get_key("probe")
    bot = VisionBot(
        rpc_url=rpc,
        vision_address=vision_addr,
        private_key=key,
        data_node_url=data_node,
        oracles=oracles,
    )

    min_deposit = bot.vision.functions.MIN_DEPOSIT().call()
    print(f"MIN_DEPOSIT: {min_deposit} wei ({min_deposit / 1e18} USDC)")
    print(f"USDC address (self-discovered): {bot.usdc_address}")
    next_batch_id = bot.vision.functions.nextBatchId().call()
    print(f"nextBatchId: {next_batch_id}")
    print(f"bot address: {bot.bot_addr}")

    try:
        r = _retry_get(f"{data_node}/health", timeout=10)
        print(f"data-node /health: status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        print(f"data-node /health: ERROR {e}")

    for o in oracles:
        try:
            r = requests.get(f"{o}/", timeout=10)
            print(f"oracle {o}: status={r.status_code}")
        except requests.RequestException as e:
            print(f"oracle {o}: ERROR {e}")

    sys.exit(0)


def cmd_dryrun(args):
    rpc = cfg("RPC_URL")
    vision_addr = cfg("VISION_ADDRESS")
    data_node = cfg("DATA_NODE_URL").rstrip("/")
    oracles = [u.strip().rstrip("/") for u in cfg("ORACLE_URLS").split(",") if u.strip()]

    key = get_key("dryrun")
    bot = VisionBot(
        rpc_url=rpc,
        vision_address=vision_addr,
        private_key=key,
        data_node_url=data_node,
        oracles=oracles,
    )

    source = bot.discover_source(args.source)
    print(f"displayName: {source.get('displayName')}")
    print(f"sourceId: {source.get('sourceId')}")
    print(f"tickDurationSecs: {source.get('tickDurationSecs')}")
    print(f"data-node configHash (advisory): {source.get('configHash')}")
    markets = source.get("markets") or []
    print(f"market count: {len(markets)}")
    print(f"first 3 markets: {json.dumps(markets[:3], indent=2, default=str)}")

    data_node_hash = source["configHash"]
    config_hash = bytes.fromhex(data_node_hash[2:] if data_node_hash.startswith("0x") else data_node_hash)
    tick = int(source.get("tickDurationSecs") or 0)

    batch_id, config_hash = bot.find_active_batch_id(config_hash, tick_duration=tick)
    on_chain_hex = "0x" + config_hash.hex()
    print(f"batch_id: {batch_id}  on-chain configHash (used in tx): {on_chain_hex}")
    if on_chain_hex.lower() != data_node_hash.lower():
        print(
            "  (data-node hash differs — mid-rotation race. "
            "tx uses the on-chain value, per invariant 7.)"
        )
    batch = bot.get_batch(batch_id)
    printable = dict(batch)
    for k in ("source_id", "config_hash"):
        v = printable[k]
        if isinstance(v, (bytes, bytearray)):
            printable[k] = "0x" + bytes(v).hex()
    print(f"getBatch({batch_id}): {json.dumps(printable, default=str)}")

    n = len(markets)
    snapshot_by_id = bot.fetch_snapshot(args.source)
    features_df = None
    markets_by_id = {m["assetId"]: m for m in markets if m.get("assetId")}
    if args.strategy in FEATURE_STRATEGIES:
        from features import extract_features
        from history import fetch_history

        asset_ids = list(markets_by_id.keys())
        print(f"fetching {args.history_hours}h history for {len(asset_ids)} assets…")
        hist = fetch_history(
            data_node_url=data_node,
            asset_ids=asset_ids,
            hours=args.history_hours,
        )
        print(f"  got {len(hist)} rows covering {hist['asset_id'].nunique() if not hist.empty else 0} assets")
        features_df = extract_features(
            hist,
            snapshot_by_id=snapshot_by_id,
            markets_by_id=markets_by_id,
        )
        predictor = make_predictor_with_features(
            args.strategy,
            features_df=features_df,
            xgb_model_path=args.xgb_model,
            claude_top_k=args.claude_top_k,
            markets_by_id=markets_by_id,
        )
    else:
        predictor = make_predictor(args.strategy)
    scores = predictor.predict(markets, snapshot_by_id)
    picks = picks_from_scores(scores, threshold=args.threshold)

    summary = pick_summary(picks)
    print(
        f"strategy: {predictor.name}  threshold={args.threshold}  "
        f"snapshot_rows={len(snapshot_by_id)}  picks={summary}"
    )
    # Show the 5 strongest signals in each direction so the user
    # can sanity-check what the bot is actually betting on.
    ranked = sorted(
        zip(markets, scores), key=lambda t: t[1], reverse=True
    )
    strongest_up = [(m["assetId"], s) for m, s in ranked[:5] if s > args.threshold]
    strongest_down = [(m["assetId"], s) for m, s in ranked[-5:] if s <= args.threshold]
    print(f"top UP   : {strongest_up}")
    print(f"top DOWN : {strongest_down}")

    bitmap = encode_bitmap(picks, n)
    bmhash = hash_bitmap(bitmap)
    print(f"bitmap length: {len(bitmap)}")
    print(f"bitmap_hash: 0x{bmhash.hex()}")

    deposit_wei = int(args.deposit * 1e18)
    print(f"deposit_wei: {deposit_wei}")

    tx = bot.build_join_tx(batch_id, config_hash, deposit_wei, bmhash)
    serializable = {}
    for k, v in tx.items():
        if isinstance(v, (bytes, bytearray)):
            serializable[k] = "0x" + bytes(v).hex()
        else:
            serializable[k] = v
    print(f"joinBatchDirect tx: {json.dumps(serializable, indent=2, default=str)}")
    sys.exit(0)


def cmd_trade(args):
    rpc = cfg("RPC_URL")
    vision_addr = cfg("VISION_ADDRESS")
    data_node = cfg("DATA_NODE_URL").rstrip("/")
    oracles = [u.strip().rstrip("/") for u in cfg("ORACLE_URLS").split(",") if u.strip()]

    key = get_key("trade")
    bot = VisionBot(
        rpc_url=rpc,
        vision_address=vision_addr,
        private_key=key,
        data_node_url=data_node,
        oracles=oracles,
    )

    source = bot.discover_source(args.source)
    markets = source.get("markets") or []
    n = len(markets)
    config_hash_hex = source["configHash"]
    config_hash = bytes.fromhex(config_hash_hex[2:] if config_hash_hex.startswith("0x") else config_hash_hex)
    tick = int(source.get("tickDurationSecs") or 0)

    batch_id, config_hash = bot.find_active_batch_id(config_hash, tick_duration=tick)
    print(f"Active batch {batch_id} with {n} markets (on-chain configHash=0x{config_hash.hex()})")

    deposit_wei = int(args.deposit * 1e18)
    snapshot_by_id = bot.fetch_snapshot(args.source)
    features_df = None
    markets_by_id = {m["assetId"]: m for m in markets if m.get("assetId")}
    if args.strategy in FEATURE_STRATEGIES:
        from features import extract_features
        from history import fetch_history

        hist = fetch_history(
            data_node_url=data_node,
            asset_ids=list(markets_by_id.keys()),
            hours=args.history_hours,
        )
        features_df = extract_features(
            hist,
            snapshot_by_id=snapshot_by_id,
            markets_by_id=markets_by_id,
        )
        predictor = make_predictor_with_features(
            args.strategy,
            features_df=features_df,
            xgb_model_path=args.xgb_model,
            claude_top_k=args.claude_top_k,
            markets_by_id=markets_by_id,
        )
    else:
        predictor = make_predictor(args.strategy)
    scores = predictor.predict(markets, snapshot_by_id)
    picks = picks_from_scores(scores, threshold=args.threshold)
    summary = pick_summary(picks)
    print(f"strategy: {predictor.name}  picks={summary}")
    bitmap = encode_bitmap(picks, n)
    bmhash = hash_bitmap(bitmap)

    print("Approving USDC…")
    bot.approve_usdc(deposit_wei)
    print("Joining batch…")
    tx_hash = bot.join_batch(batch_id, config_hash, deposit_wei, bmhash)
    print(f"joined tx={tx_hash.hex()}")
    print("Submitting bitmap to oracles…")
    accepted = bot.submit_bitmap(batch_id, bitmap, bmhash)
    print(f"oracle accepted: {accepted}/{len(oracles)}")

    print("Waiting for settlement…")
    while True:
        payout = bot.get_payout(batch_id)
        if payout > 0:
            pnl = (payout - deposit_wei) / 1e18
            print(f"Settled. PnL = {pnl:+.4f} USDC")
            break
        time.sleep(15)
    sys.exit(0)


def cmd_train_xgb(args):
    from history import fetch_history
    from xgb_predictor import XGBPredictor

    data_node = cfg("DATA_NODE_URL").rstrip("/")
    rpc = cfg("RPC_URL")
    vision_addr = cfg("VISION_ADDRESS")
    oracles = [u.strip().rstrip("/") for u in cfg("ORACLE_URLS").split(",") if u.strip()]

    key = get_key("train-xgb") if False else Account.create().key.hex()
    bot = VisionBot(
        rpc_url=rpc, vision_address=vision_addr,
        private_key=key, data_node_url=data_node, oracles=oracles,
    )
    source = bot.discover_source(args.source)
    markets = source.get("markets") or []
    markets_by_id = {m["assetId"]: m for m in markets if m.get("assetId")}
    all_ids = list(markets_by_id.keys())
    asset_ids = all_ids[: args.max_assets] if args.max_assets else all_ids
    print(f"Training XGB on {len(asset_ids)} assets × {args.hours}h history…")

    hist = fetch_history(data_node_url=data_node, asset_ids=asset_ids, hours=args.hours)
    if hist.empty:
        print("No history returned. Nothing to train on.")
        sys.exit(2)
    print(f"  got {len(hist)} rows over {hist['asset_id'].nunique()} assets")

    predictor = XGBPredictor()
    stats = predictor.train(
        hist,
        save_path=args.out,
        markets_by_id={
            k: v for k, v in markets_by_id.items() if k in asset_ids
        },
    )
    print(f"Training done: {stats}")


def cmd_backtest(args):
    from history import fetch_history
    from backtest import walk_forward

    data_node = cfg("DATA_NODE_URL").rstrip("/")
    rpc = cfg("RPC_URL")
    vision_addr = cfg("VISION_ADDRESS")
    oracles = [u.strip().rstrip("/") for u in cfg("ORACLE_URLS").split(",") if u.strip()]

    bot = VisionBot(
        rpc_url=rpc, vision_address=vision_addr,
        private_key=Account.create().key.hex(),
        data_node_url=data_node, oracles=oracles,
    )
    source = bot.discover_source(args.source)
    markets = source.get("markets") or []
    markets_by_id = {m["assetId"]: m for m in markets if m.get("assetId")}
    asset_ids = list(markets_by_id.keys())[: args.max_assets]
    print(f"Backtesting {args.strategy} on {len(asset_ids)} assets × {args.hours}h history…")

    hist = fetch_history(data_node_url=data_node, asset_ids=asset_ids, hours=args.hours)
    if hist.empty:
        print("No history returned. Cannot backtest.")
        sys.exit(2)

    def factory():
        if args.strategy in FEATURE_STRATEGIES:
            return make_predictor_with_features(
                args.strategy,
                features_df=None,  # walk_forward builds features per tick
                xgb_model_path=args.xgb_model,
            )
        return make_predictor(args.strategy)

    stats = walk_forward(
        hist,
        factory,
        markets_by_id={k: v for k, v in markets_by_id.items() if k in asset_ids},
    )
    print(f"Backtest: {stats}")


def cmd_gen_keys(args):
    env_path = Path(__file__).parent / ".env"
    env_text = env_path.read_text() if env_path.exists() else ""
    new_lines = []
    for tag in ("A", "B"):
        var = f"BOT_PRIVATE_KEY_{tag}"
        if f"\n{var}=" in "\n" + env_text and not env_text.rstrip().endswith(f"{var}="):
            # Already present with value — skip. Print the address.
            line = next(
                (l for l in env_text.splitlines() if l.startswith(f"{var}=") and len(l) > len(var) + 2),
                None,
            )
            if line:
                key = line.split("=", 1)[1].strip()
                addr = Account.from_key(key).address
                print(f"{var} already set — address: {addr}")
                continue
        acct = Account.create()
        key = acct.key.hex()
        if not key.startswith("0x"):
            key = "0x" + key
        new_lines.append(f"{var}={key}")
        print(f"{var} generated — address: {acct.address}")
    if new_lines:
        sep = "" if env_text.endswith("\n") or not env_text else "\n"
        env_path.write_text(env_text + sep + "\n".join(new_lines) + "\n")
        print(f"wrote {len(new_lines)} keys to {env_path}")


def cmd_fund(args):
    rpc = cfg("RPC_URL")
    vision_addr = cfg("VISION_ADDRESS")
    key = os.getenv("BOT_PRIVATE_KEY")
    if not key:
        print("BOT_PRIVATE_KEY required (the funding wallet)")
        sys.exit(2)
    bot = VisionBot(
        rpc_url=rpc, vision_address=vision_addr, private_key=key,
        data_node_url=cfg("DATA_NODE_URL"), oracles=[],
    )
    amount_wei = int(args.amount * 1e18)
    bal = bot.usdc.functions.balanceOf(bot.bot_addr).call()
    if bal < amount_wei:
        print(f"insufficient balance: have {bal/1e18:.4f}, need {args.amount}")
        sys.exit(3)
    to = Web3.to_checksum_address(args.to)
    tx = bot.usdc.functions.transfer(to, amount_wei).build_transaction({
        "from": bot.bot_addr,
        "nonce": bot.w3.eth.get_transaction_count(bot.bot_addr),
        "gas": 120_000,
        "gasPrice": bot.w3.eth.gas_price,
        "chainId": bot.w3.eth.chain_id,
    })
    signed = bot.account.sign_transaction(tx)
    raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
    h = bot.w3.eth.send_raw_transaction(raw)
    receipt = bot.w3.eth.wait_for_transaction_receipt(h, timeout=120)
    print(f"transfer {args.amount} USDC → {to}: tx {h.hex()} status {receipt.status}")


# Public anvil account index 1. Not a secret — this is the standard
# Hardhat/Foundry test-mnemonic key ("test test test ... junk") and is
# widely used to seed L3 / devnet contracts. On this testnet it holds
# ~260 ETH native and can call the public mint() on the L3 WUSDC.
# Re-running faucet many times is safe: it only adds supply; other
# operators can also mint. Do NOT use on mainnet.
_ANVIL_FAUCET_KEY = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"


def cmd_faucet(args):
    rpc = cfg("RPC_URL")
    vision_addr = cfg("VISION_ADDRESS")
    faucet_acct = Account.from_key(_ANVIL_FAUCET_KEY)

    bot = VisionBot(
        rpc_url=rpc, vision_address=vision_addr,
        private_key=_ANVIL_FAUCET_KEY,
        data_node_url=cfg("DATA_NODE_URL"), oracles=[],
    )
    to = Web3.to_checksum_address(args.to)

    def _send_raw(tx: dict, label: str) -> str:
        signed = faucet_acct.sign_transaction(tx)
        raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
        h = bot.w3.eth.send_raw_transaction(raw)
        r = bot.w3.eth.wait_for_transaction_receipt(h, timeout=60)
        hh = h.hex()
        short = hh[:18]
        print(f"  {label}: tx {short}… status {r.status}")
        return hh

    # Native gas
    if args.eth > 0:
        nonce = bot.w3.eth.get_transaction_count(faucet_acct.address)
        _send_raw({
            "from": faucet_acct.address, "to": to, "value": int(args.eth * 1e18),
            "nonce": nonce, "gas": 21000, "gasPrice": bot.w3.eth.gas_price,
            "chainId": bot.w3.eth.chain_id,
        }, f"{args.eth} ETH → {to}")

    # Mint L3 USDC
    if args.usdc > 0:
        mint_abi = [{
            "inputs": [{"name": "to", "type": "address"},
                       {"name": "amount", "type": "uint256"}],
            "name": "mint", "outputs": [],
            "stateMutability": "nonpayable", "type": "function",
        }]
        usdc = bot.w3.eth.contract(address=bot.usdc.address, abi=mint_abi)
        nonce = bot.w3.eth.get_transaction_count(faucet_acct.address)
        tx = usdc.functions.mint(to, int(args.usdc * 1e18)).build_transaction({
            "from": faucet_acct.address, "nonce": nonce,
            "gas": 120_000, "gasPrice": bot.w3.eth.gas_price,
            "chainId": bot.w3.eth.chain_id,
        })
        _send_raw(tx, f"mint {args.usdc} USDC → {to}")

    eth_bal = bot.w3.eth.get_balance(to) / 1e18
    usdc_bal = bot.usdc.functions.balanceOf(to).call() / 1e18
    print(f"  final: {to}  ETH {eth_bal:.4f}  USDC {usdc_bal:.4f}")


def cmd_balance(args):
    from dotenv import dotenv_values

    rpc = cfg("RPC_URL")
    vision_addr = cfg("VISION_ADDRESS")
    env = dotenv_values(Path(__file__).parent / ".env")
    keys_found = {}
    for k, v in env.items():
        if k.startswith("BOT_PRIVATE_KEY") and v:
            try:
                keys_found[k] = Account.from_key(v)
            except Exception:
                pass
    if not keys_found:
        print("No BOT_PRIVATE_KEY* entries in .env")
        sys.exit(2)

    bot = VisionBot(
        rpc_url=rpc,
        vision_address=vision_addr,
        private_key=next(iter(keys_found.values())).key.hex(),
        data_node_url=cfg("DATA_NODE_URL"),
        oracles=[],
    )
    print(f"{'var':<22} {'address':<44} {'USDC':>12}")
    for var, acct in keys_found.items():
        try:
            bal = bot.usdc.functions.balanceOf(acct.address).call()
        except Exception as e:
            print(f"{var:<22} {acct.address:<44} ERR {e}")
            continue
        print(f"{var:<22} {acct.address:<44} {bal/1e18:>12.4f}")


def main():
    parser = argparse.ArgumentParser(prog="vision-bot")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("probe")

    for sub_parser in []:
        pass

    def _add_strategy_args(p):
        p.add_argument(
            "--strategy",
            default="momentum",
            choices=ALL_STRATEGIES,
        )
        p.add_argument("--threshold", type=float, default=0.0)
        p.add_argument(
            "--history-hours",
            type=int,
            default=6,
            help="Hours of history to fetch for feature-based strategies.",
        )
        p.add_argument(
            "--xgb-model",
            default=os.getenv("XGB_MODEL_PATH", "models/xgb.pkl"),
            help="Path to trained XGBoost model (for --strategy xgb/ensemble/claude).",
        )
        p.add_argument(
            "--claude-top-k",
            type=int,
            default=20,
            help="How many marginal picks Claude gets to override.",
        )

    p_dry = sub.add_parser("dryrun")
    p_dry.add_argument("--source", default="twitch")
    p_dry.add_argument("--deposit", type=float, default=0.1)
    _add_strategy_args(p_dry)

    p_trade = sub.add_parser("trade")
    p_trade.add_argument("--source", default="twitch")
    p_trade.add_argument("--deposit", type=float, required=True)
    _add_strategy_args(p_trade)

    p_train = sub.add_parser("train-xgb")
    p_train.add_argument("--source", default="twitch")
    p_train.add_argument(
        "--hours",
        type=int,
        default=168,
        help="Hours of history to pull (data-node retains ~24 days).",
    )
    p_train.add_argument("--out", default="models/xgb.pkl")
    p_train.add_argument(
        "--max-assets",
        type=int,
        default=0,
        help="Cap on assets (0 = use all).",
    )

    p_keys = sub.add_parser(
        "gen-keys",
        help="Generate BOT_PRIVATE_KEY_A/B for side-by-side racing. "
        "Appends to .env if not already present; prints addresses.",
    )

    p_fund = sub.add_parser(
        "fund", help="Transfer L3 USDC from BOT_PRIVATE_KEY to an address."
    )
    p_fund.add_argument("--to", required=True, help="Recipient L3 address")
    p_fund.add_argument("--amount", type=float, required=True, help="USDC amount")

    p_fct = sub.add_parser(
        "faucet",
        help="Seed a wallet from the L3 testnet faucet (mint USDC + send gas). "
        "Uses the public anvil account index 1 key — valid only on this testnet.",
    )
    p_fct.add_argument("--to", required=True, help="Recipient L3 address")
    p_fct.add_argument("--usdc", type=float, default=1.0, help="USDC amount to mint")
    p_fct.add_argument("--eth", type=float, default=0.01, help="Native ETH to send for gas")

    p_bal = sub.add_parser(
        "balance", help="Print L3 USDC balance of each configured wallet."
    )

    p_bt = sub.add_parser("backtest")
    p_bt.add_argument("--source", default="twitch")
    p_bt.add_argument("--hours", type=int, default=6)
    p_bt.add_argument(
        "--strategy",
        default="rolling",
        choices=ALL_STRATEGIES,
    )
    p_bt.add_argument(
        "--xgb-model",
        default=os.getenv("XGB_MODEL_PATH", "models/xgb.pkl"),
    )
    p_bt.add_argument(
        "--max-assets",
        type=int,
        default=100,
        help="Cap on assets used for backtest (full 8k is slow).",
    )

    args = parser.parse_args()
    if args.cmd == "probe":
        cmd_probe(args)
    elif args.cmd == "dryrun":
        cmd_dryrun(args)
    elif args.cmd == "trade":
        cmd_trade(args)
    elif args.cmd == "train-xgb":
        cmd_train_xgb(args)
    elif args.cmd == "backtest":
        cmd_backtest(args)
    elif args.cmd == "gen-keys":
        cmd_gen_keys(args)
    elif args.cmd == "fund":
        cmd_fund(args)
    elif args.cmd == "faucet":
        cmd_faucet(args)
    elif args.cmd == "balance":
        cmd_balance(args)


if __name__ == "__main__":
    main()
