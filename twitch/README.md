# twitch — Vision-testnet trading bot

Python bot that trades the Twitch source on Vision testnet (Index L3, chainId 111222333).

Current batch: **~8200 markets**, 92% streamers (`twitch_stream_*`), 8% games (`twitch_game_*`). Tick **60 s**. MIN_DEPOSIT **0.1 L3 USDC**.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# optionally set BOT_PRIVATE_KEY in .env for `trade`
# set ANTHROPIC_API_KEY in .env for `--strategy claude`
```

Python 3.11+ required (3.14 recommended). `brew install libomp` on macOS for xgboost.

## Pipeline

```
  snapshot ────┐
               │
  history ──→ features ──→ predictor ──→ scores ──→ picks ──→ bitmap ──→ joinBatchDirect
               │                                     (threshold)           (1024 bytes, keccak)
  snapshot ────┘
```

Every strategy conforms to the same contract:

```python
predictor.predict(markets, snapshot_by_id) -> list[float]
# positive → UP likely, negative → DOWN likely
```

`picks_from_scores(scores, threshold=0.0)` binarises. Trading transport stays strategy-agnostic.

## Commands

```bash
# Sanity-check infra — no credentials, no cost.
python main.py probe

# Dryrun: build the join tx with a real strategy, print it, exit.
python main.py dryrun --strategy rolling --history-hours 2

# Train XGBoost on N assets × H hours of real history (~24 days retained).
python main.py train-xgb --hours 168 --max-assets 500 --out models/xgb.pkl

# Walk-forward backtest: score every historical tick, compare to actuals.
python main.py backtest --strategy rolling --hours 6 --max-assets 100

# Real trade — signs and sends.
python main.py trade --strategy ensemble --deposit 0.1
```

## Strategy stack

| `--strategy` | Uses history? | Uses ML? | Uses Claude? | What it does |
|---|:---:|:---:|:---:|---|
| `all_yes` / `all_no` | no | no | no | Baselines. |
| `momentum` | no | no | no | `changePct / 100`. Snapshot only. |
| `contrarian` | no | no | no | `-changePct / 100`. |
| `logistic` | no | no | no | `tanh(0.5 × changePct)`. Confidence-weighted momentum. |
| `rolling` | **yes** | no | no | Weighted sum of rolling 5m/15m/1h/6h/24h changes, shaped by streak length. |
| `xgb` | yes | **yes** | no | XGBoost binary classifier — features from `features.extract_features`, labels from next-tick direction. Regularised + early-stopping. |
| `claude` | yes | delegated | **yes** | Wraps a base predictor; for the top-K markets where the base is uncertain (scores near threshold), calls Claude with the asset's rolling summary and lets it override. |
| `ensemble` | yes | yes | optional | Weighted blend. Default: 50% rolling + 50% xgb. |

## Feature set (`features.extract_features`)

Per asset, one row:

- `change_5m`, `change_15m`, `change_1h`, `change_6h`, `change_24h` — rolling pct change from value at window-start.
- `vol_1h`, `vol_24h` — std of pct-change returns.
- `mean_1h`, `mean_24h` — mean value over window.
- `slope_1h` — linear-regression slope, normalised (%/hour).
- `streak` — length of current same-sign change run (positive = up).
- `n_obs_24h` — sample count. Directly usable as confidence gate.
- `current_change_pct` — live snapshot value (may be ahead of the last history row).

## History access

Data-node endpoint: `GET /market/batch-history?assets=id1,id2,...&from=ISO`.

- Max 100 assets per request (server-enforced).
- Retention: ~24 days.
- Density varies — popular assets get updates every minute or two; long-tail assets get a handful per day.

`history.fetch_history(assets, hours)` chunks concurrently (8 workers) and returns one DataFrame.

```
300 assets × 168h →  757k rows in ~128s
8192 assets × 2h → 189k rows in  ~37s
```

## Claude integration

`claude_predictor.ClaudePredictor` wraps any base predictor. For the top-K markets whose base scores sit closest to the threshold, it builds a compact prompt per marginal pick (assetId, name, current change, rolling features) and asks Claude for a single UP/DOWN verdict. Overrides the base score only on those marginals.

Cost scales with `--claude-top-k` (default 20), not market count. Zero tokens on markets where the base is already confident.

```bash
export ANTHROPIC_API_KEY=sk-...
python main.py dryrun --strategy claude --claude-top-k 30
```

## Adding your own predictor

Three lines in `strategy.py`:

```python
class MyPredictor:
    name = "my"
    def predict(self, markets, snapshot_by_id):
        return [some_math(m, snapshot_by_id.get(m["assetId"])) for m in markets]

REGISTRY["my"] = MyPredictor
```

If it needs features, include `make_predictor_with_features` handling for the name.

## Walk-forward backtest

```bash
python main.py backtest --strategy rolling --hours 6 --max-assets 100
```

For each (asset, tick_i) with enough history, features are computed from ticks `[0..i]`, the predictor scores the asset, the score is binarised, and the prediction is compared to `sign(value[i+1] - value[i])`. Output:

```
{'n': 14237, 'accuracy': 0.51, 'log_loss': 0.69, 'direction_up_rate': 0.48}
```

If your accuracy isn't comfortably above `direction_up_rate` and its complement, you have no edge — the strategy is just tracking drift.

## Notes

- Bitmap is always 1024 bytes, MSB-first per byte. Market `i` → bit `7 - (i % 8)` of byte `i // 8`.
- L3 USDC has 18 decimals. `int(0.1 * 1e18)` for 0.1 USDC.
- Pool totals are not queryable. You trade blind — the edge comes from your predictor.
- `probe` and `dryrun` auto-generate an ephemeral key if `BOT_PRIVATE_KEY` is unset.
- USDC address is self-discovered via `vision.functions.USDC().call()`.
