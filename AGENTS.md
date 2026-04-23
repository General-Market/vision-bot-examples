# AGENTS.md

Bootstrap instructions for a fresh agent picking up this repo. If you're reading this to rebuild from scratch — follow this file literally.

## What this repo is

A standalone trading bot for the Twitch source on Vision testnet (Index L3, chainId `111222333`). Trades a ~8200-market parimutuel every 60 seconds. The model is XGBoost on short-horizon features with resolution-aware labels; a Claude-wrapped variant overrides marginal picks.

The transport (bitmap encoding, `joinBatchDirect`, oracle quorum reveal, `PlayerSettled` settlement) must be byte-exact or the oracle silently rejects your deposit. The strategy is a swappable predictor: `momentum`, `rolling`, `xgb`, `ensemble`, `claude`.

## Zero-to-trading — fully unattended

```bash
git clone https://github.com/General-Market/vision-bot-examples
cd vision-bot-examples/twitch
./setup.sh --auto-fund        # ≈ 2 min; wallet is seeded from the L3 faucet
.venv/bin/python live_trader.py --strategy momentum --deposit 0.1
```

`setup.sh` is idempotent. With `--auto-fund` it:

1. Detects Python ≥ 3.11 (macOS system 3.9 won't work — install Homebrew's `python@3.14`).
2. Creates `.venv`, installs deps (macOS: `brew install libomp` if xgboost needed).
3. Generates `BOT_PRIVATE_KEY` into `.env` if missing.
4. Calls `main.py faucet --to $BOT_ADDR --usdc 1 --eth 0.01` — seeds the wallet from the L3 testnet faucet (public anvil account index 1 with mint rights on L3 WUSDC).

Without `--auto-fund`, the script prints the address and tells the user to fund it themselves.

Model training is NOT in setup. The live loop runs `momentum`/`rolling`/`contrarian`/`all_yes` out of the box — all feature-derived, zero model artifact needed. For `xgb`/`ensemble`/`claude`, run:

```bash
.venv/bin/python main.py train-xgb --hours 72 --max-assets 500 --out models/xgb.pkl
```

≈ 3 min. Produces a model with real edge (~0.5 pp lift, ~20 % flip-catch).

## Then trade

```bash
# Sanity check — no funds required.
.venv/bin/python main.py probe

# Dry run — builds the tx, prints it, doesn't sign.
.venv/bin/python main.py dryrun --strategy ensemble

# Live loop — signs, sends, reveals, logs PnL every 60s.
.venv/bin/python live_trader.py --deposit 0.1 --strategy ensemble

# In another terminal:
.venv/bin/python -c 'from pnl_logger import report; report("pnl.db")'
```

The loop is crash-resilient: any RPC hiccup, oracle timeout, or bad tick is logged and skipped. Ctrl-C exits cleanly after the current iteration.

## Race two strategies

The point of `live_trader.py` is to measure real PnL, not offline accuracy. The repo ships a race harness:

```bash
# Generate two keys — appends BOT_PRIVATE_KEY_A/B to .env, prints addresses.
.venv/bin/python main.py gen-keys

# Fund both addresses with L3 USDC. If you have a primary funded wallet,
# use the fund subcommand:
BOT_PRIVATE_KEY=<funded-key> .venv/bin/python main.py fund \
    --to 0xB168EF…  --amount 0.5
BOT_PRIVATE_KEY=<funded-key> .venv/bin/python main.py fund \
    --to 0xcbc70b…  --amount 0.5

# Check balances:
.venv/bin/python main.py balance

# Start the race:
./race.sh ensemble all_yes 0.1

# Watch live PnL:
.venv/bin/python race_report.py pnl-ensemble.db pnl-all_yes.db
```

`race.sh` spawns both `live_trader.py` processes in the background with separate DBs and logs under `logs/`. Ctrl-C stops both. `race_report.py` prints joins, settled, cumulative PnL, wagered, ROI, and the head-to-head difference.

This is the only way to know whether the ML edge survives real parimutuel competition. Offline accuracy numbers (97%) are mostly stickiness; the real edge is ~0.5–1.3 pp of lift over the naive baseline, and whether that translates to USDC depends on who else is betting.

## Public infrastructure (no VPN)

| | URL |
|---|---|
| L3 RPC | `http://142.132.164.24/` |
| Chain ID | `111222333` |
| Vision contract | `0x94d540bb45975bd5a0c7ba9a15a0d34e378f6c61` |
| L3 WUSDC | self-discovered via `vision.USDC()` |
| Data-node (cached) | `https://generalmarket.io/bot-api` |
| Oracles (3) | `http://116.203.156.98/oracle{1,2,3}` |

The data-node path is a Varnish cache on VPS 3. Retail access to the origin data-node is firewalled — only the cache can reach it. Rate limit on the public path: 60 requests per minute per IP.

## Key files

```
twitch/
  bitmap.py               encode_bitmap + hash_bitmap (1024-byte padded keccak)
  vision_bot.py           on-chain client: discover_source, find_active_batch_id,
                          approve_usdc, join_batch, submit_bitmap, get_payout
  history.py              bulk per-asset price history (data-node)
  features.py             rolling features + resolution distance
  xgb_predictor.py        vectorised training + inference
  strategy.py             predictor registry; make_predictor_with_features
  claude_predictor.py     Claude-as-tiebreaker wrapper
  ensemble.py             weighted blend
  backtest.py             walk-forward with flip/stuck split
  live_trader.py          production loop
  pnl_logger.py           SQLite ledger + report()
  race.sh                 spawn two live_traders side by side
  race_report.py          compare DBs, print head-to-head PnL
  main.py                 CLI: probe, dryrun, trade, train-xgb, backtest
  setup.sh                one-shot bootstrap
  abi/Vision.json         full Foundry ABI
  abi/ERC20.json          minimal approve/balanceOf
```

## The invariants — do not break

1. Bitmap is always **1024 bytes**, MSB-first. Short buffers hash wrong → oracle rejects → deposit lost.
2. L3 USDC has **18 decimals**. `int(0.1 * 10**18)`. Never `1e6`.
3. `MIN_DEPOSIT = 1e17 wei = 0.1 USDC`. Lower joins revert.
4. Bitmap reveal must follow the on-chain join confirmation. Revealing before = 404.
5. `config_hash` from the on-chain `getBatch` is the one that goes into `joinBatchDirect` — not the data-node's recommended hash (they can differ during config rotation).
6. USDC address is self-discovered via `vision.functions.USDC().call()`. The JSON files in the mono repo are stale.
7. Pool totals are not queryable. You trade blind — edge comes from your model, not the book.
8. Miss a tick rather than burn gas on a guaranteed revert. If `now + 10s > tick_end - lock_offset`, skip.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `getBatch failed: could not transact` | Vision address stale | Run `main.py probe`; confirm bytecode length > 0. Update `VISION_ADDRESS` from `/batches/recommended` response if needed. |
| `data-node status=429` | Rate-limited on public cache | Throttle `FEATURE_REFRESH_SEC` up (default 900). Cache TTLs mean feature refreshes rarely need to hit origin. |
| `Training set too small (0 rows)` | Too few assets × hours | Raise `--hours` or `--max-assets`. At least ~5000 rows needed. |
| `97% training accuracy` | Probably leakage | Check FEATURE_COLS does not include `current_resolution`. |
| `xgboost library could not be loaded` | macOS, no libomp | `brew install libomp` |
| `TypeError: unsupported operand type(s) for |` | Python < 3.11 | Switch interpreter; see setup.sh |

## License

MIT.
