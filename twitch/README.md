# twitch-vision-bot

A trading bot for Twitch stream outcome markets on Vision testnet (Index L3 Orbit chain, chainId `111222333`). Combines three probability layers — external streaming analytics, Vision on-chain parimutuel pool prices, and a custom ML model with Claude API for interpretation.

Batch: `twitch` — `batchId=19`, `tickDuration=60s`.

## Prerequisites

- Python 3.11+ (tested on 3.14)
- macOS users: `brew install libomp` (xgboost depends on OpenMP)
- A Twitch developer app (client_id + app access token)
- An Anthropic API key
- Optional: a funded testnet wallet for on-chain betting

## Setup

```bash
git clone <repo> && cd twitch-vision-bot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: TWITCH_CLIENT_ID, TWITCH_APP_TOKEN, ANTHROPIC_API_KEY
# leave BOT_PRIVATE_KEY blank unless you want the bot to actually trade
```

## Usage

```bash
# Smoke test: verify imports and config
python -m pytest tests/ -v

# List active Twitch markets on Vision (read-only, no creds needed)
python main.py markets

# Predict a single market (needs Twitch creds + Anthropic key)
python main.py predict --channel xqc --question "Will xqc exceed 40k peak viewers tonight?"

# Run the full daily pipeline (load → features → train → predict)
python main.py pipeline
```

## Architecture

| Layer | Purpose |
|---|---|
| `data/` | Twitch Helix, SullyGnome scrape, cleaning, binary labeling |
| `features/` | Rolling stats, Four Factors, ELO, fatigue, overlap, triple-layer divergence |
| `vision/` | Index L3 RPC client, historical batch prices, order submission |
| `models/` | LR, RF, XGBoost, soft-voting ensemble, hybrid predictor |
| `evaluation/` | Walk-forward backtest, ablation, metrics |
| `visualization/` | Confusion matrix, calibration, divergence scatter |

## L3 Decimals

Vision batch pools use **18-decimal USDC** (L3 wrapped). Settlement-chain USDC is 6-decimal. Do not mix them. All pool-size conversions in this bot divide by `1e18`.

## Known limitations

- The Vision ABI exposes batch metadata (`getBatch`, `currentTickId`, `joinBatchDirect`) but no YES/NO pool totals — parimutuel legs are off-chain in the bitmap layer. `get_market_price` returns the canonical shape but defaults pool values to 0.5/0.5 until the off-chain bitmap reader is integrated.
- `VisionTrader.submit_bet(side, amount)` raises `NotImplementedError`. The only member-entry function in the current ABI is `joinBatchDirect(batchId, configHash, amount, bitmapHash)`, which requires the pick-encoding hash to be computed upstream. Use `VisionTrader.join_batch(...)` once you have the hashes.
- If `main.py markets` prints `getBatch(N) failed: ... is contract deployed correctly`, the `VISION_ADDRESS` in your `.env` points to an address with no bytecode on the configured RPC. Update `VISION_ADDRESS` from the active deployment, or point `VISION_RPC_URL` at a chain where the contract is live.

## Testnet Safety

Do not run with a non-empty `BOT_PRIVATE_KEY` unless the key holds testnet-only funds. The bot will submit real transactions to the configured RPC.

## Retail-user smoke test (fresh clone)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -v
# Expected: 6 passed, 1 skipped (network test needs credentials)
```
