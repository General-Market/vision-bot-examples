import type { AssetDoc, PricePoint, Trade } from './types'
import { aiDecide, makeActualBetLookup } from './strategies'

export type Range = '1h' | '1d' | '1w' | '1m' | 'all'

const RANGE_SECONDS: Record<Range, number> = {
  '1h': 60 * 60,
  '1d': 24 * 60 * 60,
  '1w': 7 * 24 * 60 * 60,
  '1m': 30 * 24 * 60 * 60,
  all: Number.POSITIVE_INFINITY,
}

export const RANGE_LABEL: Record<Range, string> = {
  '1h': '1H',
  '1d': '1D',
  '1w': '1W',
  '1m': '1M',
  all: 'ALL',
}

export const RANGES: Range[] = ['1h', '1d', '1w', '1m', 'all']

/**
 * Slice an asset's history+trades to the trailing window. Falls back to
 * the full doc if the window leaves us with too few points to render.
 */
export function sliceByRange(doc: AssetDoc, range: Range): AssetDoc {
  if (range === 'all') return doc
  if (doc.history.length === 0) return doc
  const last = doc.history[doc.history.length - 1].ts
  const cutoff = last - RANGE_SECONDS[range]
  const history = doc.history.filter((p) => p.ts >= cutoff)
  if (history.length < 2) return doc
  const trades = (doc.trades ?? []).filter((t) => t.joined_at >= cutoff)
  return { ...doc, history, trades }
}

export interface AiBacktest {
  final: number
  wins: number
  losses: number
  ticks: number
}

/**
 * Replays the AI strategy across the price history and returns the
 * cumulative score plus a win/loss tally. The strategy is the same one
 * PnLChart graphs — kept here as a pure number for the hero stat.
 */
export function aiBacktest(history: PricePoint[]): AiBacktest {
  let cum = 0
  let wins = 0
  let losses = 0
  let ticks = 0
  for (let i = 0; i + 1 < history.length; i++) {
    const bet = aiDecide(i, history)
    if (!bet) continue
    const moveUp = history[i + 1].price > history[i].price
    const moveDown = history[i + 1].price < history[i].price
    if (!moveUp && !moveDown) continue
    ticks++
    const win = (bet === 'UP' && moveUp) || (bet === 'DOWN' && moveDown)
    if (win) {
      cum++
      wins++
    } else {
      cum--
      losses++
    }
  }
  return { final: cum, wins, losses, ticks }
}

export interface ActualPerf {
  wins: number
  losses: number
  open: number
  pnl: number
}

export function actualPerf(trades: Trade[]): ActualPerf {
  let wins = 0
  let losses = 0
  let open = 0
  let pnl = 0
  for (const t of trades) {
    if (!t.settled) {
      open++
      continue
    }
    const p = t.pnl_usdc ?? 0
    pnl += p
    if (p > 0) wins++
    else if (p < 0) losses++
  }
  return { wins, losses, open, pnl }
}

/**
 * Re-uses the actual-bet lookup for marker rendering elsewhere, without
 * pulling strategies into chart components directly.
 */
export const lookupActualBet = makeActualBetLookup
