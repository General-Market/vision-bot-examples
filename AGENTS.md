# AGENTS.md

Bootstrap instructions for a fresh agent picking up this repo. If you're reading this to rebuild from scratch — follow this file literally.

## What this repo is

Two reference prediction bots for Vision testnet (Index L3, chainId `111222333`), plus a shared local visualizer.

- `twitch/` — ~8200 streamer and game viewership markets, 60 s ticks.
- `polymarket/` — Polymarket events mirrored on Vision under `poly_*`.
- `visualizer/` — React + Vite SPA that reads either bot's `pnl.json`.

Both bots are the same skeleton with a different signal universe. The model is XGBoost on short-horizon features with resolution-aware labels; a Claude-wrapped layer interprets the divergence between the three probability sources.

**These are prediction bots, not trading bots.** They read the chain, build features, train, and emit probabilities. Neither closes the loop: `VisionTrader.submit_bet` raises `NotImplementedError` because a YES/NO side cannot be turned into the bitmap commitment `joinBatchDirect` demands, and nothing here encodes bitmaps. `join_batch(...)` will sign a transaction if you compute `config_hash` and `bitmap_hash` yourself.

## Layout

Identical between the two bots:

```
<source>/
  config/settings.py      all runtime config, env-overridable — SOURCE OF TRUTH
                          for RPC, chain id, contract addresses, data-node URL
  data/                   source loader, cleaner, telemetry, binary labelling
  features/               rolling stats, four factors, ELO, fatigue, overlap,
                          triple-layer divergence, Claude context features
  models/                 LR / RF / XGBoost, soft-voting ensemble, hybrid
                          predictor, prepare_model_data + train_and_evaluate
  vision/                 read-only L3 client, historical batch prices, trader
  evaluation/             walk-forward backtest, ablation, metrics
  visualization/          confusion matrix, calibration, divergence scatter
  abi/Vision.json         full Foundry ABI
  main.py                 CLI: markets, predict, pipeline
  pipeline.py             the daily orchestration behind `main.py pipeline`
  tests/test_basic.py     smoke tests — 6 pass, 1 skips without credentials
  requirements.txt
  .env.example            mirrors config/settings.py defaults
```

`features/triple_layer.py` is byte-identical between the two bots. It is pure math and carries no domain logic. Only `data/` and the per-feature modules differ.

## Bootstrap

```bash
git clone https://github.com/General-Market/vision-bot-examples
cd vision-bot-examples
./setup.sh                                  # defaults to twitch
./setup.sh --source polymarket              # the sibling bot
```

`setup.sh` is idempotent. It:

1. Finds a Python ≥ 3.11 on PATH (macOS system 3.9 will not work — `brew install python@3.14`).
2. Creates `.venv` at the repo root and installs `<source>/requirements.txt`.
3. Warns if xgboost cannot load — on macOS that means `brew install libomp`.
4. Copies `<source>/.env.example` to `<source>/.env` if none exists. An existing `.env` is never touched.

Then, from inside the bot directory:

```bash
cd twitch
../.venv/bin/python -m pytest tests/ -v      # expect: 6 passed, 1 skipped
../.venv/bin/python main.py markets          # read-only, no credentials
../.venv/bin/python main.py predict --channel xqc \
    --question "Will xqc exceed 40k peak viewers tonight?"
../.venv/bin/python main.py pipeline         # load -> features -> train -> predict
```

`markets` needs nothing. `predict` and `pipeline` need the source credentials in `.env` (`TWITCH_CLIENT_ID` + `TWITCH_APP_TOKEN` for twitch; nothing for polymarket, whose Gamma and CLOB read endpoints are public) plus `ANTHROPIC_API_KEY` for the Claude layer.

Leave `BOT_PRIVATE_KEY` blank. With it blank both bots are strictly read-only and `VisionTrader` returns `{"status": "dryrun", ...}` instead of signing.

## Configuration — one source of truth

`<source>/config/settings.py` holds every default. `.env` overrides it. `.env.example` mirrors it. When they disagree, **settings.py wins and the others are the bug.**

| | Value | Env var |
|---|---|---|
| L3 RPC | `http://142.132.164.24/` | `VISION_RPC_URL` |
| Chain ID | `111222333` | `VISION_CHAIN_ID` |
| Vision contract | `0x80Ab4ebDF79dEa442b54DECdcEd16D6654470544` | `VISION_ADDRESS` |
| Index contract | `0xaBf79086293d30C8A72A0BE700a1c492F0Dd9D3a` | `INDEX_ADDRESS` |
| L3 WUSDC | `0x2710e49EBb807A0cB9369F13Ba24Bd809809a827` | `L3_USDC_ADDRESS` |
| Data-node | `https://api.generalmarket.io` | `DATA_NODE_URL` |
| Twitch batch | `19` (tick 60 s) | `TWITCH_BATCH_ID` |
| Polymarket batch | `0` = discover at runtime | `POLYMARKET_BATCH_ID` |

`visualizer/download.py` carries its own copy of the RPC and data-node values in `DEFAULTS`. Keep it in step with `settings.py`.

## The invariants — do not break

1. **L3 USDC has 18 decimals.** `int(0.1 * 10**18)`. Never `1e6`. Every pool conversion here divides by `1e18`.
2. **`MIN_DEPOSIT = 1e17 wei = 0.1 USDC`.** Lower joins revert.
3. **The pick is a 1024-byte bitmap**, zero-padded, MSB-first, hashed to a commitment. Short buffers hash wrong — the batch keeps your deposit and records no pick. This repo does not encode bitmaps; you must.
4. **`config_hash` comes from the on-chain `getBatch`**, not from the data-node's `/batches/recommended`. They diverge during config rotation, and only the on-chain value is accepted.
5. **The USDC address is self-discoverable** via `vision.functions.USDC().call()`. Prefer it over any committed JSON when the two disagree — the chain is truth.
6. **Pool totals are not queryable.** The ABI exposes `getBatch`, `currentTickId` and `joinBatchDirect` but no YES/NO totals — the parimutuel legs live off-chain in the bitmap layer. `get_market_price` returns the canonical shape with pools defaulted to 0.5/0.5 until an off-chain bitmap reader exists.

## Visualizer

```bash
cd visualizer
pnpm install                                  # Node 20+, pnpm 9+
python download.py --pnl ../twitch/pnl.json --player 0xYourBotAddress
pnpm dev                                      # http://localhost:5173
pnpm typecheck && pnpm build
```

`download.py` writes `public/index.json` and `public/data/<batch_id>/<asset_id>.json` from the public data-node. Override the endpoints with `--data-node` / `--oracle` if you run your own node.

## Honest status

Every accuracy number in this repo — the 97%, the ~0.5 pp lift — is offline classification against historical data. Most of that 97% is stickiness: the naive "same as last tick" baseline is already very strong on viewership markets. No wallet has signed a real `joinBatchDirect` and observed `PlayerSettled`, so no claim about real parimutuel PnL is supported by anything here.

Whether the offline edge survives real competition is unmeasured. Do not report it as though it were.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'models'` | Old clone, whose `.gitignore` excluded `models/` source | Pull latest. The root `.gitignore` now ignores `*.pkl` artifacts, never the package. |
| `TypeError: unsupported operand type(s) for \|` | Python < 3.11 | Use 3.11+; `setup.sh` picks the newest it finds. |
| `xgboost library could not be loaded` | macOS, no OpenMP | `brew install libomp` |
| `getBatch(N) failed: ... is contract deployed correctly` | `VISION_ADDRESS` has no bytecode on the configured RPC | Confirm `eth_getCode` is non-empty. Re-point `VISION_ADDRESS` or `VISION_RPC_URL`. |
| `data-node status=429` | Rate-limited on the public cache | Back off. Cache TTLs mean feature refreshes rarely need the origin. |
| `Training set too small (0 rows)` | Too few assets × lookback | Raise `LOOKBACK_DAYS` / `MAX_EVENTS_PER_CATEGORY`. Roughly 5000 rows needed. |
| `97% training accuracy` | Probably leakage | Check the feature set does not carry `current_resolution`. |

## License

MIT.
