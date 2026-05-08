"""Runtime configuration resolved from environment."""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BOT_ROOT = Path(__file__).resolve().parent.parent
ABI_DIR = BOT_ROOT / "abi"


def _csv(var: str, default: str = "") -> list[str]:
    raw = os.getenv(var, default).strip()
    return [x.strip() for x in raw.split(",") if x.strip()]


@dataclass
class PolymarketConfig:
    """Polymarket Gamma + CLOB endpoints. Public, no auth needed for reads."""
    gamma_url: str = os.getenv(
        "POLYMARKET_GAMMA_URL", "https://gamma-api.polymarket.com"
    )
    clob_url: str = os.getenv(
        "POLYMARKET_CLOB_URL", "https://clob.polymarket.com"
    )
    categories: list[str] = field(
        default_factory=lambda: _csv(
            "POLYMARKET_CATEGORIES", "politics,sports,crypto,world"
        )
    )
    lookback_days: int = int(os.getenv("LOOKBACK_DAYS", "120"))
    # Hard cap on events fetched per category. The full archive is large.
    max_events_per_category: int = int(
        os.getenv("MAX_EVENTS_PER_CATEGORY", "500")
    )


@dataclass
class VisionConfig:
    rpc_url: str = os.getenv("VISION_RPC_URL", "http://142.132.164.24/")
    chain_id: int = int(os.getenv("VISION_CHAIN_ID", "111222333"))
    index_address: str = os.getenv(
        "INDEX_ADDRESS", "0xaBf79086293d30C8A72A0BE700a1c492F0Dd9D3a"
    )
    vision_address: str = os.getenv(
        "VISION_ADDRESS", "0x80Ab4ebDF79dEa442b54DECdcEd16D6654470544"
    )
    usdc_address: str = os.getenv(
        "L3_USDC_ADDRESS", "0x2710e49EBb807A0cB9369F13Ba24Bd809809a827"
    )
    # Resolved at runtime from /batches/recommended?source=polymarket. The
    # sentinel 0 means "discover, don't trust this default."
    polymarket_batch_id: int = int(os.getenv("POLYMARKET_BATCH_ID", "0"))
    bot_private_key: str = os.getenv("BOT_PRIVATE_KEY", "")
    # L3 USDC uses 18 decimals. Never assume 6 on L3.
    usdc_decimals: int = 18


@dataclass
class ModelConfig:
    rolling_window: int = int(os.getenv("ROLLING_WINDOW", "10"))
    threshold_quantile: float = float(
        os.getenv("THRESHOLD_QUANTILE", "0.5")
    )
    initial_train_size: int = int(os.getenv("INITIAL_TRAIN_SIZE", "1000"))
    walkforward_step: int = int(os.getenv("WALKFORWARD_STEP", "25"))


@dataclass
class Config:
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    data_node_url: str = os.getenv(
        "DATA_NODE_URL", "https://api.generalmarket.io"
    )
    BOT_ROOT: Path = field(default_factory=lambda: BOT_ROOT)
    ABI_DIR: Path = field(default_factory=lambda: ABI_DIR)


config = Config()
