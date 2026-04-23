# vision-bot-examples

Reference trading bots for [Vision](https://generalmarket.io) — an on-chain parimutuel prediction-market layer running on Index L3 (Arbitrum Orbit, chainId `111222333`).

Each subdirectory is a standalone bot for one Vision source. **When no source is specified, `twitch` is the default.** Bring a testnet wallet, pick a strategy, trade.

## Examples

| Path | Source | Markets | Tick | Notes |
|---|---|---|---|---|
| [`twitch/`](./twitch) | `twitch` | ~8200 streamer + game viewership markets | 60 s | XGBoost + Claude + rolling ensemble |

More sources incoming — polymarket mirror, crypto price thresholds, weather, others. The transport is source-agnostic; only the strategy layer cares.

## Zero-to-trading in under 3 minutes

```bash
git clone https://github.com/General-Market/vision-bot-examples
cd vision-bot-examples
./setup.sh --auto-fund                                       # defaults to twitch
.venv/bin/python twitch/live_trader.py --deposit 0.1 --max-joins 1
```

The root `./setup.sh` forwards to `twitch/setup.sh` when no source is given. Pass `--source <name>` or set `SOURCE=<name>` when other bots ship.

Full bootstrap + trading flow is documented in [`AGENTS.md`](./AGENTS.md) — the canonical guide, including wallet funding, model training, two-wallet racing, and troubleshooting.

## Live testnet infrastructure

| | URL |
|---|---|
| L3 RPC | `http://142.132.164.24/` |
| Chain ID | `111222333` |
| Vision contract | `0x94d540bb45975bd5a0c7ba9a15a0d34e378f6c61` |
| L3 WUSDC | self-discover via `vision.USDC()` |
| Data-node (cached) | `https://generalmarket.io/bot-api` |
| Oracles (3) | `http://116.203.156.98/oracle{1,2,3}` |

All reachable over public HTTP — no VPN, no credentials for reads.

## Protocol invariants

1. **Bitmap is always 1024 bytes.** Padded with zero bits, MSB-first. Short bitmaps hash to the wrong commitment and the oracle rejects — your deposit sits in the pool with no pick.
2. **L3 USDC is 18 decimals.** Not 6. Don't copy-paste Ethereum mainnet math.
3. **Minimum deposit is 0.1 USDC** (`1e17` wei). Lower joins revert.
4. **Pool totals are not queryable.** Bots trade blind — edge comes from the predictor, not the pool.
5. **Bitmap reveal to oracles must follow the on-chain join.** Reveal first, rejected.
6. **Deployment JSON files can lag.** The chain is truth. Check `eth_getCode(vision_address)` before trusting a hard-coded address.
7. **Data-node config vs on-chain config can differ during tick rotation.** Only the value returned by `getBatch(batch_id)` goes into `joinBatchDirect` — never the one from `/batches/recommended` when they disagree.

## PnL status

**Not yet measured.** Every accuracy / lift number in this repo is offline classification against historical data (97% overall, ~0.5 pp over the sticky baseline). No wallet has signed a real `joinBatchDirect` and observed `PlayerSettled`. To produce a real PnL number, see the racing harness in `AGENTS.md` — `race.sh` spawns two funded wallets on two strategies and `race_report.py` diffs the cumulative PnL.

## Security

- **Never commit `.env`.** Each example ships with `.env.example` — copy, fill locally.
- Testnet funds only. Don't reuse a mainnet key.
- The bot's only on-chain action is `approve` + `joinBatchDirect` against the configured Vision contract. No arbitrary external calls.

## License

MIT.
