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
from vision_bot import VisionBot, _retry_get

load_dotenv(Path(__file__).parent / ".env")

DEFAULTS = {
    "RPC_URL": "http://142.132.164.24/",
    "VISION_ADDRESS": "0x94d540bb45975bd5a0c7ba9a15a0d34e378f6c61",
    "DATA_NODE_URL": "http://116.203.156.98/data-node",
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
    print(f"configHash: {source.get('configHash')}")
    print(f"sourceId: {source.get('sourceId')}")
    print(f"tickDurationSecs: {source.get('tickDurationSecs')}")
    markets = source.get("markets") or []
    print(f"market count: {len(markets)}")
    print(f"first 3 markets: {json.dumps(markets[:3], indent=2, default=str)}")

    config_hash_hex = source["configHash"]
    config_hash = bytes.fromhex(config_hash_hex[2:] if config_hash_hex.startswith("0x") else config_hash_hex)

    batch_id = bot.find_active_batch_id(config_hash)
    print(f"batch_id: {batch_id}")
    batch = bot.get_batch(batch_id)
    printable = dict(batch)
    for k in ("source_id", "config_hash"):
        v = printable[k]
        if isinstance(v, (bytes, bytearray)):
            printable[k] = "0x" + bytes(v).hex()
    print(f"getBatch({batch_id}): {json.dumps(printable, default=str)}")

    n = len(markets)
    picks = ["UP"] * n
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

    batch_id = bot.find_active_batch_id(config_hash)
    print(f"Active batch {batch_id} with {n} markets")

    deposit_wei = int(args.deposit * 1e18)
    picks = ["UP"] * n
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


def main():
    parser = argparse.ArgumentParser(prog="vision-bot")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("probe")

    p_dry = sub.add_parser("dryrun")
    p_dry.add_argument("--source", default="twitch")
    p_dry.add_argument("--deposit", type=float, default=0.1)

    p_trade = sub.add_parser("trade")
    p_trade.add_argument("--source", default="twitch")
    p_trade.add_argument("--deposit", type=float, required=True)

    args = parser.parse_args()
    if args.cmd == "probe":
        cmd_probe(args)
    elif args.cmd == "dryrun":
        cmd_dryrun(args)
    elif args.cmd == "trade":
        cmd_trade(args)


if __name__ == "__main__":
    main()
