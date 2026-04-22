# vision-bot-examples

Reference trading bots for [Vision](https://generalmarket.io) — an on-chain parimutuel prediction-market layer running on Index L3 (Arbitrum Orbit, chainId `111222333`).

Each subdirectory is a standalone bot for one Vision source. Bring a testnet wallet, pick a strategy, trade.

## Examples

| Path | Source | Markets | Tick | Notes |
|---|---|---|---|---|
| [`twitch/`](./twitch) | `twitch` | ~8200 streamer + game viewership markets | 60 s | XGBoost + Claude + rolling ensemble |

More sources incoming — polymarket mirror, crypto price thresholds, weather, others. The transport is source-agnostic; only the strategy layer cares.

## Quick start

```bash
cd twitch
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # macOS: brew install libomp

python main.py probe                                         # verify infra
python main.py dryrun --strategy momentum --deposit 0.1      # build tx, don't sign
python main.py dryrun --strategy xgb --history-hours 2       # with ML
python main.py trade  --strategy ensemble --deposit 0.1      # sign and send
```

## Live testnet infrastructure

| | URL |
|---|---|
| L3 RPC | `http://142.132.164.24/` |
| Chain ID | `111222333` |
| Vision contract | `0x94d540bb45975bd5a0c7ba9a15a0d34e378f6c61` |
| L3 WUSDC | self-discover via `vision.USDC()` |
| Data-node | `http://116.203.156.98/data-node` |
| Oracles (3) | `http://116.203.156.98/oracle{1,2,3}` |

All reachable over public HTTP — no VPN, no credentials for reads.

## Protocol invariants

1. **Bitmap is always 1024 bytes.** Padded with zero bits, MSB-first. Short bitmaps hash to the wrong commitment and the oracle rejects — your deposit sits in the pool with no pick.
2. **L3 USDC is 18 decimals.** Not 6. Don't copy-paste Ethereum mainnet math.
3. **Minimum deposit is 0.1 USDC** (`1e17` wei). Lower joins revert.
4. **Pool totals are not queryable.** Bots trade blind — edge comes from the predictor, not the pool.
5. **Bitmap reveal to oracles must follow the on-chain join.** Reveal first, rejected.
6. **Deployment JSON files can lag.** The chain is truth. Check `eth_getCode(vision_address)` before trusting a hard-coded address.

## Security

- **Never commit `.env`.** Each example ships with `.env.example` — copy, fill locally.
- Testnet funds only. Don't reuse a mainnet key.
- The bot's only on-chain action is `approve` + `joinBatchDirect` against the configured Vision contract. No arbitrary external calls.

## License

MIT.
