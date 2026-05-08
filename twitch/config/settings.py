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
class TwitchConfig:
    client_id: str = os.getenv("TWITCH_CLIENT_ID", "")
    app_token: str = os.getenv("TWITCH_APP_TOKEN", "")
    channels: list[str] = field(
        default_factory=lambda: _csv("CHANNELS", "xqc,kai_cenat")
    )
    lookback_days: int = int(os.getenv("LOOKBACK_DAYS", "120"))


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
    twitch_batch_id: int = int(os.getenv("TWITCH_BATCH_ID", "19"))
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
    twitch: TwitchConfig = field(default_factory=TwitchConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    data_node_url: str = os.getenv(
        "DATA_NODE_URL", "https://api.generalmarket.io"
    )
    BOT_ROOT: Path = field(default_factory=lambda: BOT_ROOT)
    ABI_DIR: Path = field(default_factory=lambda: ABI_DIR)


config = Config()
