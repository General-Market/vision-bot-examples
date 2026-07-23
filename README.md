# vision-bot-examples

Reference prediction bots for [Vision](https://generalmarket.io) — an on-chain parimutuel prediction-market layer running on Index L3 (Arbitrum Orbit, chainId `111222333`).

Each subdirectory is a standalone bot for one Vision source. **When no source is specified, `twitch` is the default.** Bring a testnet wallet, pick a source, run the pipeline.

## Examples

| Path | Source | Markets | Tick | Notes |
|---|---|---|---|---|
| [`twitch/`](./twitch) | `twitch` | ~8200 streamer + game viewership markets | 60 s | XGBoost + Claude + soft-voting ensemble |
| [`polymarket/`](./polymarket) | `polymarket` | Polymarket events mirrored on Vision (`poly_*`) | 60 s | Same skeleton as `twitch/`. Different signal. |
| [`visualizer/`](./visualizer) | — | shared local web app | — | React + Vite. Reads any bot's `pnl.json`, renders dashboard, asset list, price + PnL charts. |

Both bots share one layout: `data/` → `features/` → `models/` → `evaluation/`, with `vision/` as the read-only chain client. `features/triple_layer.py` is identical between them — pure math, source-agnostic.

More sources incoming. The transport is source-agnostic; only the strategy layer cares.

## Zero to predictions

```bash
git clone https://github.com/General-Market/vision-bot-examples
cd vision-bot-examples
./setup.sh                                  # defaults to twitch
cd twitch
../.venv/bin/python -m pytest tests/ -v     # 6 passed, 1 skipped
../.venv/bin/python main.py markets         # read-only, no credentials
```

`./setup.sh` creates `.venv`, installs that bot's requirements, and copies `.env.example` to `.env`. Pass `--source polymarket` (or `SOURCE=polymarket`) for the sibling bot. It needs Python 3.11+; macOS system Python 3.9 will not work.

Each bot's CLI is `main.py` with three subcommands:

| Command | Needs credentials | Does |
|---|---|---|
| `main.py markets` | no | lists active Vision markets for that source |
| `main.py predict` | yes | one market, one prediction, all three layers |
| `main.py pipeline` | yes | full daily run: load → features → train → predict |

Per-bot detail lives in [`twitch/README.md`](./twitch/README.md) and [`polymarket/README.md`](./polymarket/README.md). Repo-wide bootstrap and troubleshooting is [`AGENTS.md`](./AGENTS.md).

## Live testnet infrastructure

Defaults are set in each bot's `config/settings.py` — **that file is the single source of truth.** Every value below is overridable through `.env`; `.env.example` mirrors it.

| | Value |
|---|---|
| L3 RPC (`VISION_RPC_URL`) | `http://142.132.164.24/` |
| Chain ID (`VISION_CHAIN_ID`) | `111222333` |
| Vision contract (`VISION_ADDRESS`) | `0x80Ab4ebDF79dEa442b54DECdcEd16D6654470544` |
| Index contract (`INDEX_ADDRESS`) | `0xaBf79086293d30C8A72A0BE700a1c492F0Dd9D3a` |
| L3 WUSDC (`L3_USDC_ADDRESS`) | `0x2710e49EBb807A0cB9369F13Ba24Bd809809a827`, or self-discover via `vision.USDC()` |
| Data-node (`DATA_NODE_URL`) | `https://api.generalmarket.io` |

All reachable over public HTTP — no VPN, no credentials for reads.

## Protocol invariants

1. **L3 USDC is 18 decimals.** Not 6. Don't copy-paste Ethereum mainnet math.
2. **Minimum deposit is 0.1 USDC** (`1e17` wei). Lower joins revert.
3. **Pool totals are not queryable.** Bots predict blind — edge comes from the model, not the book.
4. **`joinBatchDirect` takes a bitmap hash, not a side.** The pick is a 1024-byte bitmap, zero-padded, MSB-first, hashed to a commitment. Short buffers hash wrong and the batch rejects your pick while keeping your deposit. These bots do **not** encode bitmaps — `VisionTrader.join_batch(...)` requires you to supply `config_hash` and `bitmap_hash` yourself.
5. **Deployment JSON files can lag.** The chain is truth. Check `eth_getCode(vision_address)` before trusting a hard-coded address.
6. **Data-node config vs on-chain config can differ during tick rotation.** Only the value returned by `getBatch(batch_id)` goes into `joinBatchDirect` — never the one from `/batches/recommended` when they disagree.

## Status — read this before believing a number

**No PnL has been measured.** Every accuracy and lift figure in this repo is offline classification against historical data (~97% overall, ~0.5 pp over the sticky baseline). Most of that 97% is stickiness, not skill.

No wallet in this repo has signed a real `joinBatchDirect` and observed `PlayerSettled`. `VisionTrader.submit_bet(side, amount)` deliberately raises `NotImplementedError`: a raw YES/NO side cannot be turned into the bitmap commitment the contract wants. `join_batch(...)` will sign, but only once you compute the hashes upstream.

These are prediction bots with a chain client attached. They are not yet trading bots.

## Security

- **Never commit `.env`.** Each example ships with `.env.example` — copy, fill locally.
- Testnet funds only. Don't reuse a mainnet key.
- With `BOT_PRIVATE_KEY` blank, both bots are strictly read-only. The only on-chain action they can take is `approve` + `joinBatchDirect` against the configured Vision contract. No arbitrary external calls.

## License

MIT.
