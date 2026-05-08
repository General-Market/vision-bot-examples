import type { Bet, PricePoint, Trade } from './types'

/**
 * Momentum + mean-reversion ensemble with a vol regime filter.
 * Calm market → trust momentum. Choppy → fade to the mean.
 */
export function aiDecide(i: number, prices: PricePoint[]): Bet | null {
  if (i < 30) return null
  const cur = prices[i].price
  const past20 = prices[i - 20].price

  let sum = 0
  for (let k = i - 29; k <= i; k++) sum += prices[k].price
  const ma30 = sum / 30

  let vsum = 0
  let vcount = 0
  for (let k = i - 29; k < i; k++) {
    const r = (prices[k + 1].price - prices[k].price) / Math.max(1e-9, prices[k].price)
    vsum += r * r
    vcount++
  }
  const vol = Math.sqrt(vsum / Math.max(1, vcount))
  const momSign = cur > past20 * 1.001 ? 1 : cur < past20 * 0.999 ? -1 : 0
  const revSign = cur > ma30 * 1.01 ? -1 : cur < ma30 * 0.99 ? 1 : 0
  const w = vol > 0.02 ? 0.3 : 0.7
  const score = w * momSign + (1 - w) * revSign

  if (score > 0.25) return 'UP'
  if (score < -0.25) return 'DOWN'
  return null
}

export interface StratContext {
  actualBet: (ts: number) => Bet | null
}

export interface Strategy {
  key: 'ai' | 'actual'
  label: string
  color: string
  decide: (i: number, prices: PricePoint[], ctx: StratContext) => Bet | null
}

export const STRATEGIES: Strategy[] = [
  {
    key: 'ai',
    label: 'AI',
    color: '#ff9500',
    decide: (i, prices) => aiDecide(i, prices),
  },
  {
    key: 'actual',
    label: "bot's actual",
    color: '#1d1d1f',
    decide: (i, prices, ctx) => ctx.actualBet(prices[i].ts),
  },
]

const TICK_SECS_DEFAULT = 60

export function makeActualBetLookup(trades: Trade[]): (ts: number) => Bet | null {
  const sorted = [...trades].sort((a, b) => a.joined_at - b.joined_at)
  const tickSecs = sorted[0]?.tick_duration ?? TICK_SECS_DEFAULT
  return (ts: number) => {
    let best: Trade | null = null
    for (const t of sorted) {
      if (t.joined_at <= ts) best = t
      else break
    }
    if (!best) return null
    const window = (best.tick_duration || tickSecs) * 30
    if (ts - best.joined_at > window) return null
    return best.bet
  }
}

export function priceAt(series: PricePoint[], ts: number): number | null {
  if (!series.length) return null
  let lo = 0
  let hi = series.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (series[mid].ts < ts) lo = mid + 1
    else hi = mid
  }
  return series[lo].price
}
