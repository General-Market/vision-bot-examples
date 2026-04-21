import json
import os
import time
from pathlib import Path

import requests
from eth_account import Account
from web3 import Web3

from bitmap import encode_bitmap, hash_bitmap

ABI_DIR = Path(__file__).parent / "abi"


def _retry_get(url: str, timeout: int = 15, attempts: int = 3, backoff: float = 2.0):
    last = None
    for i in range(attempts):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code >= 500 or r.status_code == 0:
                last = r
                time.sleep(backoff)
                continue
            return r
        except requests.RequestException as e:
            last = e
            time.sleep(backoff)
    if isinstance(last, requests.Response):
        return last
    raise last if last else RuntimeError(f"GET {url} failed")


class VisionBot:
    def __init__(
        self,
        rpc_url: str,
        vision_address: str,
        private_key: str,
        data_node_url: str,
        oracles: list[str],
        usdc_address: str | None = None,
    ):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.w3.is_connected():
            raise RuntimeError(f"RPC unreachable: {rpc_url}")

        self.account = Account.from_key(private_key)
        self.bot_addr = self.account.address

        with open(ABI_DIR / "Vision.json") as f:
            vision_abi = json.load(f)["abi"]
        with open(ABI_DIR / "ERC20.json") as f:
            erc20_abi = json.load(f)["abi"]

        self.vision = self.w3.eth.contract(
            address=Web3.to_checksum_address(vision_address),
            abi=vision_abi,
        )

        # Self-discover USDC if not supplied.
        if usdc_address is None:
            usdc_address = self.vision.functions.USDC().call()
        self.usdc_address = Web3.to_checksum_address(usdc_address)
        self.usdc = self.w3.eth.contract(address=self.usdc_address, abi=erc20_abi)

        self.data_node_url = data_node_url.rstrip("/")
        self.oracles = [u.rstrip("/") for u in oracles]

    # ── read: batch metadata ──
    def get_batch(self, batch_id: int) -> dict:
        b = self.vision.functions.getBatch(batch_id).call()
        return {
            "creator":         b[0],
            "source_id":       b[1],
            "config_hash":     b[2],
            "tick_duration":   int(b[3]),
            "lock_offset":     int(b[4]),
            "created_at_tick": int(b[5]),
            "paused":          bool(b[6]),
            "settled":         bool(b[7]),
        }

    def discover_source(self, source_id: str = "twitch") -> dict:
        r = _retry_get(f"{self.data_node_url}/batches/recommended", timeout=20)
        if r.status_code != 200:
            raise RuntimeError(
                f"data-node /batches/recommended status={r.status_code} body={r.text[:300]}"
            )
        all_batches = r.json()["batches"]
        for b in all_batches:
            if b.get("sourceId") == source_id:
                return b
        raise RuntimeError(
            f"No recommended batch for sourceId={source_id!r}. "
            f"Available: {sorted({b.get('sourceId','?') for b in all_batches})}"
        )

    def fetch_snapshot(self, source_id: str = "twitch") -> dict[str, dict]:
        """Return a map assetId -> snapshot row for the named source.
        Powers the numerical predictors (changePct, value, etc.).

        `?source=` is required on the data-node — without it the endpoint
        returns a mixed-source sample of 10k rows and twitch is capped at
        ~500 (the default per-source sample ceiling).
        """
        r = _retry_get(
            f"{self.data_node_url}/vision/snapshot?source={source_id}",
            timeout=20,
        )
        if r.status_code != 200:
            raise RuntimeError(
                f"data-node /vision/snapshot status={r.status_code} body={r.text[:300]}"
            )
        rows = r.json().get("snapshots", [])
        return {row["assetId"]: row for row in rows if row.get("assetId")}

    def find_active_batch_id(
        self,
        config_hash: bytes,
        tick_duration: int | None = None,
    ) -> tuple[int, bytes]:
        """Return (batch_id, on_chain_config_hash) for the live batch.

        Prefers an exact configHash match with what the data-node proposed.
        On a rotation race (data-node has already moved to the next config
        but the chain hasn't caught up), falls back to the freshest
        unpaused/unsettled batch with the same tick_duration — and returns
        that batch's chain-recorded configHash, which is the one the join
        transaction must reference.
        """
        target = "0x" + config_hash.hex()
        same_tick: list[dict] = []
        for url in self.oracles:
            try:
                r = requests.get(f"{url}/vision/batches", timeout=10)
                if r.status_code != 200:
                    continue
                batches = r.json().get("batches", [])
                for b in batches:
                    if b.get("config_hash", "").lower() == target.lower():
                        return int(b["id"]), config_hash
                if tick_duration is not None:
                    for b in batches:
                        if (
                            int(b.get("tick_duration", -1)) == int(tick_duration)
                            and not b.get("paused", False)
                            and not b.get("settled", False)
                        ):
                            same_tick.append(b)
                if same_tick:
                    break
            except requests.RequestException:
                continue
        if not same_tick:
            raise RuntimeError(
                f"No active batch found for configHash={target} "
                f"(tick_duration={tick_duration}). The oracle's "
                f"/vision/batches did not return a match."
            )
        same_tick.sort(key=lambda b: int(b["id"]), reverse=True)
        top = same_tick[0]
        return (
            int(top["id"]),
            bytes.fromhex(top["config_hash"][2:]),
        )

    # ── tx build helpers ──
    def _build_tx(self, gas: int) -> dict:
        return {
            "from": self.bot_addr,
            "nonce": self.w3.eth.get_transaction_count(self.bot_addr),
            "gas": gas,
            "gasPrice": self.w3.eth.gas_price,
            "chainId": self.w3.eth.chain_id,
        }

    def _send(self, tx: dict) -> bytes:
        signed = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        if receipt.status != 1:
            raise RuntimeError(f"Tx reverted: {tx_hash.hex()}")
        return tx_hash

    # ── write: approve + join ──
    def approve_usdc(self, amount_wei: int) -> bytes:
        tx = self.usdc.functions.approve(
            self.vision.address, amount_wei
        ).build_transaction(self._build_tx(gas=200_000))
        return self._send(tx)

    def join_batch(
        self,
        batch_id: int,
        config_hash: bytes,
        deposit_wei: int,
        bitmap_hash: bytes,
    ) -> bytes:
        tx = self.vision.functions.joinBatchDirect(
            batch_id, config_hash, deposit_wei, bitmap_hash
        ).build_transaction(self._build_tx(gas=500_000))
        return self._send(tx)

    # ── dryrun helpers ──
    def build_join_tx(
        self,
        batch_id: int,
        config_hash: bytes,
        deposit_wei: int,
        bitmap_hash: bytes,
    ) -> dict:
        return self.vision.functions.joinBatchDirect(
            batch_id, config_hash, deposit_wei, bitmap_hash
        ).build_transaction(self._build_tx(gas=500_000))

    # ── write: reveal bitmap to oracles ──
    def submit_bitmap(
        self,
        batch_id: int,
        bitmap: bytes,
        bitmap_hash: bytes,
        timeout: float = 5.0,
    ) -> int:
        payload = {
            "player": self.bot_addr,
            "batch_id": batch_id,
            "bitmap_hex": "0x" + bitmap.hex(),
            "expected_hash": "0x" + bitmap_hash.hex(),
        }
        accepted = 0
        for url in self.oracles:
            try:
                r = requests.post(f"{url}/vision/bitmap", json=payload, timeout=timeout)
                if r.status_code == 200:
                    accepted += 1
            except requests.RequestException:
                continue
        quorum = -(-len(self.oracles) * 2 // 3)
        if accepted < quorum:
            raise RuntimeError(
                f"Oracle quorum failed: {accepted}/{len(self.oracles)} accepted, need {quorum}"
            )
        return accepted

    # ── read: settlement ──
    def get_payout(self, batch_id: int, from_block: int | None = None) -> int:
        if from_block is None:
            latest = self.w3.eth.block_number
            from_block = max(0, latest - 100_000)
        logs = self.vision.events.PlayerSettled.get_logs(
            argument_filters={"batchId": batch_id, "player": self.bot_addr},
            fromBlock=from_block,
        )
        return int(logs[-1]["args"]["payout"]) if logs else 0
