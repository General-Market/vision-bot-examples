# vision-bot (iter3)

Minimal Python trading bot for the Vision testnet on Index L3 (Arbitrum Orbit, chainId 111222333).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
# optionally set BOT_PRIVATE_KEY in .env
```

## Commands

```bash
# sanity-check infra
.venv/bin/python main.py probe

# dryrun a Twitch bet (no signing, no sending)
.venv/bin/python main.py dryrun --source twitch --deposit 0.1

# real trade (requires BOT_PRIVATE_KEY funded with L3 USDC)
.venv/bin/python main.py trade --source twitch --deposit 0.1
```

Notes:
- L3 USDC has 18 decimals.
- `probe` and `dryrun` auto-generate an ephemeral key if `BOT_PRIVATE_KEY` is unset.
- USDC address is self-discovered via `vision.USDC()`.
- Bitmap is always 1024 bytes, MSB-first per byte.
