"""Bot-initiated order submission against Vision (Index L3 Orbit)."""
from __future__ import annotations

from typing import Any

from eth_account import Account
from web3.exceptions import ContractLogicError

from config.settings import config

from .client import VisionTestnetClient


class VisionTrader:
    def __init__(
        self,
        private_key: str | None = None,
        client: VisionTestnetClient | None = None,
    ) -> None:
        self.client = client or VisionTestnetClient()
        key = (private_key or config.vision.bot_private_key or "").strip()
        if not key:
            self.account = None
            self.read_only = True
        else:
            self.account = Account.from_key(key)
            self.read_only = False

    @property
    def address(self) -> str | None:
        return self.account.address if self.account else None

    def submit_bet(
        self, batch_id: int, side: str, amount_usdc: float
    ) -> dict[str, Any]:
        if self.read_only or self.account is None:
            return {
                "status": "dryrun",
                "would_have_sent": {
                    "side": side,
                    "amount_usdc": amount_usdc,
                    "batch_id": batch_id,
                },
            }

        fn_names = self.client._fn_names
        # The Vision ABI exposes `joinBatchDirect(batchId, configHash,
        # depositAmount, bitmapHash)` as the sole member-entry function. A raw
        # YES/NO `side` is insufficient — the caller must supply configHash
        # and the pick-bitmap hash encoding `side`. Surface that gap loudly.
        if "joinBatchDirect" not in fn_names:
            raise NotImplementedError(
                "Vision ABI has no bet/submit function this client recognises. "
                f"Available functions: {sorted(fn_names)}"
            )
        raise NotImplementedError(
            "Vision uses bitmap-encoded picks via joinBatchDirect(batchId, "
            "configHash, depositAmount, bitmapHash). submit_bet(side) cannot "
            "synthesise configHash or bitmapHash from a raw YES/NO side — "
            "the pipeline must compute them from the batch's active market "
            "config before signing. See abi/Vision.json::joinBatchDirect."
        )

    # Retained for when the pipeline gains configHash + bitmapHash context.
    def join_batch(
        self,
        batch_id: int,
        config_hash: bytes,
        bitmap_hash: bytes,
        amount_usdc: float,
    ) -> dict[str, Any]:
        if self.read_only or self.account is None:
            return {
                "status": "dryrun",
                "would_have_sent": {
                    "batch_id": batch_id,
                    "amount_usdc": amount_usdc,
                },
            }
        w3 = self.client.w3
        contract = self.client.contract
        amount_raw = int(amount_usdc * 1e18)
        try:
            tx = contract.functions.joinBatchDirect(
                batch_id, config_hash, amount_raw, bitmap_hash
            ).build_transaction(
                {
                    "from": self.account.address,
                    "nonce": w3.eth.get_transaction_count(self.account.address),
                    "chainId": config.vision.chain_id,
                    "gas": 500_000,
                    "gasPrice": w3.eth.gas_price,
                }
            )
            signed = self.account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
            return {
                "tx_hash": tx_hash.hex(),
                "status": "sent",
                "amount_usdc": amount_usdc,
                "batch_id": batch_id,
            }
        except ContractLogicError as e:
            return {"status": "reverted", "error": str(e)}
