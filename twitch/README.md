# twitch — Vision-testnet trading bot

Minimal Python bot that trades the Twitch source on Vision testnet
(Index L3, chainId 111222333).

Current Twitch batch composition (~8200 markets):
- ~92% `twitch_stream_*` — individual streamer viewer markets
- ~8%  `twitch_game_*` — category/game viewership markets

Tick duration: **60 s**. MIN_DEPOSIT: **0.1 L3 USDC**.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# optionally set BOT_PRIVATE_KEY in .env
```

Python 3.11+ required. On macOS, `brew install libomp` if you later add XGBoost.

## Commands

```bash
# sanity-check infra — no credentials needed
python main.py probe

# dryrun — builds the join tx with a real strategy, prints it, exits
python main.py dryrun --source twitch --strategy momentum --deposit 0.1

# real trade — signs and sends. Requires BOT_PRIVATE_KEY funded with L3 USDC
python main.py trade --source twitch --strategy momentum --deposit 0.1
```

## Strategies

Each strategy is a **numerical predictor**: it assigns a real-valued score
to every market. The score is then binarised with a threshold to produce
the `UP`/`DOWN` pick that goes into the 1024-byte bitmap.

```
snapshot (changePct per assetId) -> predictor.predict -> [score_1, ..., score_N]
                                                          |
                                       picks_from_scores(scores, threshold)
                                                          |
                                               [UP, DOWN, UP, ..., DOWN]
                                                          |
                                                     encode_bitmap -> bytes
```

Shipped out of the box (pick with `--strategy NAME`):

| Name | What it does | Score definition |
|---|---|---|
| `all_yes` | Bets UP on everything. Baseline. | `+1` for every market |
| `all_no` | Bets DOWN on everything. Baseline. | `-1` for every market |
| `momentum` | Follows the trend. | `changePct / 100` |
| `contrarian` | Fades the trend. | `-changePct / 100` |
| `logistic` | Soft momentum, squashed through tanh. Confidence-weighted. | `tanh(0.5 * changePct)` |

The threshold is configurable:

```bash
# Only bet UP when momentum confidence is strong (> 2% change)
python main.py dryrun --strategy momentum --threshold 0.02
```

Markets with no snapshot data score 0 — the threshold decides whether
they default UP or DOWN. At `--threshold 0.0` (default), they go DOWN.

## Adding your own strategy

Subclass in `strategy.py`:

```python
class MyPredictor:
    name = "my"

    def predict(self, markets, snapshot_by_id):
        # markets: list of {"assetId": ..., "resolutionType": ..., ...}
        # snapshot_by_id: {"assetId": {"changePct": ..., "value": ..., ...}}
        # Return one float per market. Positive → UP, negative → DOWN.
        return [some_math(m, snapshot_by_id.get(m["assetId"])) for m in markets]
```

Then register it in `REGISTRY`:

```python
REGISTRY["my"] = MyPredictor
```

Run it:

```bash
python main.py dryrun --strategy my
```

The separation between **score** and **pick** is deliberate — any ML model
that outputs a per-market probability (XGBoost, logistic regression, a
neural net, a Claude synthesis score) slots in as a predictor without
touching the on-chain transport.

## Live Twitch signal

A recent dryrun surfaced these top UP candidates:

```
top UP   : twitch_stream_danislots12  +67.0
           twitch_stream_dustyz1n     +52.3
           twitch_stream_deni_koshmar +48.0
           twitch_stream_beko288      +14.0
           twitch_stream_aogofficial  +10.3
```

These are real streamers whose viewer counts were trending up at the
moment the snapshot was fetched. Whether the parimutuel agrees is
settled on the next tick boundary.

## Notes

- Bitmap is always 1024 bytes, MSB-first per byte. Market index `i` → bit `7 - (i % 8)` of byte `i // 8`.
- L3 USDC has 18 decimals. `int(0.1 * 1e18)` for 0.1 USDC.
- `probe` and `dryrun` auto-generate an ephemeral key if `BOT_PRIVATE_KEY` is unset, so infra sanity-checks cost nothing.
- USDC address is self-discovered via `vision.functions.USDC().call()`.
- Pool totals are not queryable on-chain. You trade blind — the edge comes from your predictor.
