# twitch — Vision-testnet trading bot

Trades the Twitch source on Vision testnet (Index L3, chainId 111222333).

**Current batch:** ~8200 markets (92% streamers, 8% games). Tick 60 s. MIN_DEPOSIT 0.1 L3 USDC.

Companion article: see `twitch-bot-vision-article.md` in the parent repo for the reasoning behind every design choice.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# optional: BOT_PRIVATE_KEY for `trade`, ANTHROPIC_API_KEY for `--strategy claude`
```

Python 3.11+. On macOS: `brew install libomp` (xgboost dependency).

## Commands

```bash
# Infra sanity check — no credentials needed.
python main.py probe

# Dryrun: builds the join tx with a real strategy, prints it, exits.
python main.py dryrun --strategy momentum --deposit 0.1
python main.py dryrun --strategy xgb --history-hours 2

# Train XGBoost on real history (~24 days retained upstream).
python main.py train-xgb --hours 168 --max-assets 500 --out models/xgb.pkl

# Walk-forward backtest with flip/stuck breakdown.
python main.py backtest --strategy xgb --hours 6 --max-assets 30

# Real trade — signs and sends. Requires BOT_PRIVATE_KEY funded with L3 USDC.
python main.py trade --strategy ensemble --deposit 0.1
```

## Pipeline

```
snapshot + history ─→ features ─→ predictor ─→ scores ─→ picks ─→ bitmap ─→ joinBatchDirect
                                                  (threshold)      (1024 B, keccak commitment)
```

Every predictor conforms to:

```python
predictor.predict(markets, snapshot_by_id) -> list[float]
# positive score → UP likely, negative → DOWN likely
```

`picks_from_scores(scores, threshold=0.0)` binarises. The trading transport is predictor-agnostic — swap momentum for XGBoost for Claude without touching `encode_bitmap`.

## Strategies

| `--strategy` | History? | ML? | Claude? | What it does |
|---|:---:|:---:|:---:|---|
| `all_yes` / `all_no` | — | — | — | Baselines. |
| `momentum` | — | — | — | `changePct / 100`. |
| `contrarian` | — | — | — | Fade the trend. |
| `rolling` | ✓ | — | — | Weighted short-window changes. |
| `xgb` | ✓ | ✓ | — | XGBoost classifier on resolution-aware labels. |
| `ensemble` | ✓ | ✓ | optional | Weighted blend (default 50% rolling + 50% xgb). |
| `claude` | ✓ | delegated | ✓ | Wraps a base; Claude overrides only the marginal picks. |

## Features

Short-horizon only — tick is 60 s, longer windows don't predict next-tick:

- `change_1m`, `change_5m`, `change_15m`
- `vol_5m`, `slope_5m`, `streak`, `n_obs_5m`
- `hour_utc`, `day_of_week`, `is_weekend`, `is_primetime`
- `baseline_24h` (used only as resolution-rule input, not as a change feature)
- `dist_to_up`, `dist_to_down` (signed distance to the market's threshold)
- `category_mean_5m`, `asset_vs_category_5m` (cross-asset platform signal)
- `current_change_pct`

## Labels

**The oracle resolves against each market's threshold rule, not against direction.** Training on "did the value go up" teaches the wrong question.

```python
def compute_label(next_value, baseline_24h, resolution_type, threshold_bps):
    frac = threshold_bps / 10000
    if resolution_type == "up_x":    return int(next_value > baseline_24h * (1 + frac))
    if resolution_type == "down_x":  return int(next_value < baseline_24h * (1 - frac))
    if resolution_type == "up_0":    return int(next_value > 0)
    if resolution_type == "flat_x":  return int(abs(next_value - baseline_24h) / abs(baseline_24h) < frac)
```

## Measured edge

Training on 500 assets × 168 h of history, 1.2 M rows, resolution-aware labels:

- train 97.2% / test 97.1% — no overfit
- naive "copy current resolution" baseline: 95.85%
- XGBoost: 97.16%
- **Flip-catch (the only metric that matters): ~35%** on the 4.15% of ticks that actually flip

The 1.31 pp of headline accuracy is the entire edge against bots that always copy current state.

## The invariants

1. Bitmap is always **1024 bytes**, MSB-first per byte. Shorter buffers hash wrong and reject.
2. L3 USDC is **18 decimals**. `int(0.1 * 10**18)` for 0.1 USDC.
3. MIN_DEPOSIT = 0.1 USDC (`1e17` wei). Lower joins revert.
4. `config_hash` from `getBatch` is passed verbatim to `joinBatchDirect`.
5. Bitmap reveal to oracles must follow the join tx confirmation.
6. Pool totals are not queryable. Trade blind.
7. `eth_getCode(vision_address)` must return non-empty before you trust a hard-coded address.
8. USDC address is self-discovered via `vision.USDC()`.
9. `probe` and `dryrun` auto-generate an ephemeral key if `BOT_PRIVATE_KEY` is unset.
