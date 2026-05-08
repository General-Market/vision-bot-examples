"""Read-only Vision client against Index L3 Orbit (chainId 111222333)."""
from __future__ import annotations

import json
import time
from typing import Any

from web3 import Web3

from config.settings import ABI_DIR, config


class VisionTestnetClient:
    def __init__(
        self,
        rpc_url: str | None = None,
        vision_address: str | None = None,
    ) -> None:
        self.rpc_url = rpc_url or config.vision.rpc_url
        self.vision_address = Web3.to_checksum_address(
            vision_address or config.vision.vision_address
        )
        with open(ABI_DIR / "Vision.json") as f:
            self.abi = json.load(f)["abi"]
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        if not self.w3.is_connected():
            raise ConnectionError(f"Cannot reach Vision RPC at {self.rpc_url}")
        self.contract = self.w3.eth.contract(
            address=self.vision_address, abi=self.abi
        )
        self._fn_names = {
            e["name"] for e in self.abi if e.get("type") == "function"
        }

    def list_active_markets(self, source_prefix: str = "twitch") -> list[dict]:
        # ABI has no enumerate-active-batches function; fall back to the
        # configured Twitch batch id. `source_prefix` is kept for parity with
        # the bot's expected interface.
        _ = source_prefix
        try:
            market = self.get_market_price(config.vision.twitch_batch_id)
            return [market] if market else []
        except Exception as e:
            print(f"[vision] list_active_markets failed: {e}")
            return []

    def get_market_price(self, batch_id: int) -> dict[str, Any]:
        # The Vision ABI exposes batch metadata but no YES/NO pool totals —
        # parimutuel legs are off-chain in the bitmap layer. We return the
        # canonical shape with default 0.5/0.5 and zero liquidity.
        try:
            batch = self.contract.functions.getBatch(batch_id).call()
        except Exception as e:
            print(f"[vision] getBatch({batch_id}) failed: {e}")
            return {}

        tick_duration = int(batch[3])
        created_at_tick = int(batch[5])
        try:
            current_tick = int(
                self.contract.functions.currentTickId(batch_id).call()
            )
            settles_at = int(time.time()) + tick_duration * max(
                1, current_tick - created_at_tick + 1
            ) - int(time.time()) % max(tick_duration, 1)
        except Exception:
            settles_at = int(time.time()) + tick_duration

        yes_pool = 0.0
        no_pool = 0.0
        total = yes_pool + no_pool
        if total <= 0:
            yes_prob, no_prob, liquidity = 0.5, 0.5, 0.0
        else:
            yes_prob = yes_pool / total
            no_prob = no_pool / total
            liquidity = total / 1e18

        return {
            "yes": yes_prob,
            "no": no_prob,
            "liquidity_usdc": liquidity,
            "settles_at": settles_at,
            "batch_id": batch_id,
        }
