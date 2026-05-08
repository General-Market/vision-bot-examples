# visualizer

Local web app for inspecting your bot's bets. Reads each bot's `pnl.json`,
fetches the underlying price history from the public General Market data-node,
and renders a dashboard, asset list, price charts, and PnL charts.

Source-agnostic. The same visualizer works for `twitch/`, `polymarket/`, and
any future sibling under this repo.

## Stack

React 19 · Vite · Tailwind v4 · lightweight-charts · TanStack Query.

## Setup

```bash
pnpm install
```

Node 20+. pnpm 9+.

## Refresh data

The visualizer reads static JSON from `public/`. Build it once per session:

```bash
python download.py --pnl ../twitch/pnl.json --player 0xYourBotAddress
# or for polymarket:
python download.py --pnl ../polymarket/pnl.json --player 0xYourBotAddress
```

Defaults point at the public GM data-node and oracles. Override with
`--data-node` / `--oracle` if you run your own.

`download.py` writes `public/index.json` (one row per traded asset) and
`public/data/<batch_id>/<asset_id>.json` (price series + trade meta). The Vite
app picks them up on reload.

## Develop

```bash
pnpm dev          # http://localhost:5173
pnpm typecheck
pnpm build        # produces dist/
pnpm preview      # serves dist/ on :4173
```

## What it shows

- **Dashboard** — hero card, recent assets, PnL sparkline.
- **Asset list** — every market the bot touched, sortable by PnL, settlement, source.
- **Asset view** — price chart with bet markers, deposit/balance/PnL.

The visualizer makes no judgements. It records what the bot did and what
the market did next. Whether those two things relate is the question every
bot eventually has to answer.
