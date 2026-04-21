# vision-bot-examples

Reference trading bots for [Vision](https://generalmarket.io) — the on-chain parimutuel prediction-market layer running on Index L3 (Arbitrum Orbit, chainId `111222333`).

Each subfolder is a standalone bot for one Vision source. Bring a testnet wallet, pick a strategy, and trade.

## Examples

| Path | Source | Markets | Tick |
|---|---|---|---|
| [`twitch/`](./twitch) | `twitch` | ~5600 game-viewership markets | 60s |

More sources coming soon — Polymarket mirrors, crypto price threshold, weather, and others.

## Quick start

```bash
cd twitch
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# macOS: xgboost and scikit-learn need OpenMP
# brew install libomp

python main.py probe                              # verify infra
python main.py dryrun --source twitch --deposit 0.1   # build tx, don't sign
python main.py trade --source twitch --deposit 0.1    # sign and send (needs BOT_PRIVATE_KEY)
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

All endpoints are public HTTP — no VPN, no credentials required for read access.

## Protocol invariants (read before you bet)

1. **Bitmap is always 1024 bytes.** Padded with zero bits, MSB-first per byte. Short bitmaps hash to the wrong commitment and the oracle rejects you — your deposit sits with no pick and resolves as pure loss.
2. **L3 USDC uses 18 decimals.** `int(10 * 10**18)` for 10 USDC. Don't copy-paste 6-dec math from Ethereum mainnet.
3. **Minimum deposit is 0.1 USDC** (`MIN_DEPOSIT = 1e17` wei). Lower joins revert.
4. **Pool totals are not queryable.** Bots trade blind — your edge comes from your model, not from reading the crowd.
5. **Bitmap must be POSTed to oracles after the join transaction confirms.** Submit before, and the oracle rejects with "no on-chain commitment."
6. **The deployment JSON files in the mono repo can lag.** Trust the chain — `eth_getCode(vision_address)` and `vision.USDC()` are authoritative.

## Security

- **Never commit `.env`.** Each example ships with `.env.example` — copy to `.env` and fill in your key locally.
- Testnet funds only. Do not reuse a mainnet key.
- The bot's only on-chain action is `approve` + `joinBatchDirect` against the configured Vision contract. No arbitrary external calls.

## License

MIT.
