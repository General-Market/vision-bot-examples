# polymarket-vision-bot

A trading bot for Polymarket-mirror markets on Vision testnet (Index L3 Orbit, chainId `111222333`). Combines three probability layers — Polymarket midprice as the external signal, Vision on-chain parimutuel pool prices, and a custom ML model with the Claude API for interpretation.

Source: `polymarket` · prefixes `poly_*`.

Sibling to [`twitch/`](../twitch/). Same modular shape. Different signal universe — Polymarket Gamma + CLOB instead of Twitch Helix.

## Prerequisites

- Python 3.11+ (tested on 3.14)
- macOS users: `brew install libomp` (xgboost depends on OpenMP)
- An Anthropic API key (only for `--strategy claude` paths)
- Optional: a funded testnet wallet for on-chain betting

No Polymarket auth needed. Gamma + CLOB read endpoints are public.

## Setup

```bash
git clone <repo> && cd polymarket
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: ANTHROPIC_API_KEY (optional)
# leave BOT_PRIVATE_KEY blank unless you want the bot to actually trade
```

## Usage

```bash
# Smoke test: verify imports and config
python -m pytest tests/ -v

# List active polymarket markets on Vision (read-only, no creds needed)
python main.py markets

# Predict a single market (needs ANTHROPIC_API_KEY for the Claude layer)
python main.py predict --market-id 0x... \
                       --question "Will candidate X win the primary?" \
                       --category politics

# Run the full daily pipeline (load → features → train → predict)
python main.py pipeline
```

## Architecture

| Layer | Purpose |
|---|---|
| `data/` | Polymarket Gamma loader, CLOB telemetry, cleaning, binary labeling |
| `features/` | Rolling stats, market four factors, category ELO, fatigue, overlap, triple-layer divergence |
| `vision/` | Index L3 RPC client, historical batch prices, order submission |
| `models/` | LR, RF, XGBoost, soft-voting ensemble, hybrid predictor |
| `evaluation/` | Walk-forward backtest, ablation, metrics |
| `visualization/` | Confusion matrix, calibration, divergence scatter |

The skeleton mirrors `twitch/` byte-for-byte at the layer level. `features/triple_layer.py` is identical between the two bots — pure math, source-agnostic. Only `data/` and the per-feature modules carry domain logic.

## Triple-layer prediction

Each prediction blends three signals with weights chosen by Vision pool liquidity:

1. **ML** — XGBoost trained on settled Polymarket markets, features built from Gamma metadata + category aggregates.
2. **External** — Polymarket midprice. The crowd's prior on the original venue.
3. **Vision** — on-chain parimutuel pool ratios for the same market mirrored on Index L3.

Claude reads all three plus the KL divergence and emits an `adjusted_yes / adjusted_no` with confidence. The hybrid combines all four with liquidity-aware weights — high-liquidity Vision pools get 35% weight, thin pools fall back to ML + external.

## L3 Decimals

Vision batch pools use **18-decimal USDC** (L3 wrapped). Settlement-chain USDC is 6-decimal. Do not mix them. All pool-size conversions in this bot divide by `1e18`.

## Visualizer

Bot writes `pnl.json` after every join. To inspect bets and price drift in a browser:

```bash
cd ../visualizer
pnpm install
python download.py --pnl ../polymarket/pnl.json --player 0xYourBotAddress
pnpm dev
```

Same SPA both bots feed. See [`../visualizer/README.md`](../visualizer/README.md).

## Known limitations

- The Vision ABI exposes batch metadata (`getBatch`, `currentTickId`, `joinBatchDirect`) but no YES/NO pool totals — parimutuel legs are off-chain in the bitmap layer. `get_market_price` returns the canonical shape but defaults pool values to 0.5/0.5 until the off-chain bitmap reader is integrated.
- The Polymarket Gamma `closed=true` filter returns settled markets only. The walk-forward training universe is bounded by `LOOKBACK_DAYS` and `MAX_EVENTS_PER_CATEGORY` — set both honestly to avoid silent truncation.
- Live midprice retrieval via `MarketTelemetry` rate-limits at the public CLOB throughput. For dense polling consider running your own CLOB mirror.

## Testnet Safety

Do not run with a non-empty `BOT_PRIVATE_KEY` unless the key holds testnet-only funds. The bot will submit real transactions to the configured RPC.

## Retail-user smoke test (fresh clone)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -v
# Expected: 6 passed, 1 skipped (network test gated on POLYMARKET_NETWORK_TESTS)
```
