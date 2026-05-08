import { useEffect, useRef, useState } from 'react'
import {
  AreaSeries,
  ColorType,
  CrosshairMode,
  createChart,
  createSeriesMarkers,
  type IChartApi,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type SeriesMarker,
  type Time,
  type UTCTimestamp,
} from 'lightweight-charts'
import type { AssetDoc, Bet, PricePoint, Trade } from '@/lib/types'
import { STRATEGIES, makeActualBetLookup, priceAt } from '@/lib/strategies'

interface Props {
  data: AssetDoc
  overlay: 'ai' | 'actual' | 'none'
  showCloses?: boolean
  showSettlements?: boolean
}

interface VLine {
  x: number
  kind: 'close' | 'settlement'
}

export function PriceChart({ data, overlay, showCloses = false, showSettlements = false }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesRef = useRef<ISeriesApi<'Area'> | null>(null)
  const markersRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null)
  const [vlines, setVlines] = useState<VLine[]>([])

  // Mount: build the chart once.
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const chart = createChart(container, {
      width: container.clientWidth,
      height: container.clientHeight,
      layout: {
        background: { type: ColorType.Solid, color: '#ffffff' },
        textColor: '#6e6e73',
        fontFamily:
          '"SF Pro Text", -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif',
        attributionLogo: false,
      },
      grid: {
        vertLines: { color: 'rgba(0,0,0,0.06)' },
        horzLines: { color: 'rgba(0,0,0,0.06)' },
      },
      rightPriceScale: { borderColor: 'rgba(0,0,0,0.08)' },
      timeScale: { borderColor: 'rgba(0,0,0,0.08)', timeVisible: true, secondsVisible: false },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: 'rgba(0,0,0,0.25)', width: 1, style: 0 },
        horzLine: { color: 'rgba(0,0,0,0.25)', width: 1, style: 0 },
      },
      autoSize: false,
    })
    const series = chart.addSeries(AreaSeries, {
      lineColor: '#0071e3',
      topColor: 'rgba(0,113,227,0.18)',
      bottomColor: 'rgba(0,113,227,0)',
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: false,
    })
    const markers = createSeriesMarkers(series, [])
    chartRef.current = chart
    seriesRef.current = series
    markersRef.current = markers

    const resize = () => {
      if (!container || !chartRef.current) return
      chartRef.current.applyOptions({
        width: container.clientWidth,
        height: container.clientHeight,
      })
    }
    const ro = new ResizeObserver(resize)
    ro.observe(container)

    return () => {
      ro.disconnect()
      chart.remove()
      chartRef.current = null
      seriesRef.current = null
      markersRef.current = null
    }
  }, [])

  // Data + markers update whenever the asset or overlay changes.
  useEffect(() => {
    const series = seriesRef.current
    const markers = markersRef.current
    const chart = chartRef.current
    if (!series || !markers || !chart) return

    const points = dedupeAndSort(data.history).map((p) => ({
      time: p.ts as UTCTimestamp,
      value: p.price,
    }))
    series.setData(points)

    markers.setMarkers(buildMarkers(data, overlay))
    chart.timeScale().fitContent()
  }, [data, overlay])

  // Vertical hairlines for closes / settlements. Computed in DOM space
  // because Lightweight Charts has no native vertical-line primitive.
  useEffect(() => {
    const chart = chartRef.current
    const container = containerRef.current
    if (!chart || !container) return

    const recompute = () => {
      if (!showCloses && !showSettlements) {
        setVlines([])
        return
      }
      const scale = chart.timeScale()
      const sorted = dedupeAndSort(data.history)
      const tsToX = makeTimeToX(scale, sorted)
      const out: VLine[] = []
      const seen = new Set<string>()
      const trades = data.trades ?? []
      for (const t of trades) {
        if (showCloses) {
          const ts = t.joined_at + (t.tick_duration || 0)
          const key = `c:${ts}`
          if (!seen.has(key)) {
            seen.add(key)
            const x = tsToX(ts)
            if (x != null) out.push({ x, kind: 'close' })
          }
        }
        if (showSettlements) {
          const ts = t.joined_at + 2 * (t.tick_duration || 0)
          const key = `s:${ts}`
          if (!seen.has(key)) {
            seen.add(key)
            const x = tsToX(ts)
            if (x != null) out.push({ x, kind: 'settlement' })
          }
        }
      }
      setVlines(out)
    }

    // Initial run + a deferred run because timeToCoordinate isn't ready
    // until the chart has laid out after setData/fitContent.
    recompute()
    const raf = requestAnimationFrame(recompute)
    const scale = chart.timeScale()
    scale.subscribeVisibleTimeRangeChange(recompute)
    scale.subscribeVisibleLogicalRangeChange(recompute)
    const ro = new ResizeObserver(recompute)
    ro.observe(container)
    return () => {
      cancelAnimationFrame(raf)
      scale.unsubscribeVisibleTimeRangeChange(recompute)
      scale.unsubscribeVisibleLogicalRangeChange(recompute)
      ro.disconnect()
    }
  }, [data, showCloses, showSettlements])

  return (
    <div className="absolute inset-0">
      <div ref={containerRef} className="absolute inset-0" />
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        {vlines.map((l, i) => (
          <div
            key={i}
            className="absolute top-0 bottom-6 w-px"
            style={{
              left: l.x,
              background:
                l.kind === 'close'
                  ? 'rgba(110,110,115,0.45)'
                  : 'rgba(255,59,48,0.55)',
              borderLeft:
                l.kind === 'close'
                  ? '1px dashed rgba(110,110,115,0.55)'
                  : 'none',
            }}
          />
        ))}
      </div>
    </div>
  )
}

/**
 * Lightweight Charts' timeToCoordinate returns null for any timestamp
 * that doesn't match a data-point time exactly. For close/settlement
 * markers (synthetic timestamps offset from joined_at) we interpolate
 * between the surrounding two data points.
 */
function makeTimeToX(
  scale: ReturnType<IChartApi['timeScale']>,
  sorted: PricePoint[],
): (ts: number) => number | null {
  if (sorted.length < 2) {
    return (ts) => scale.timeToCoordinate(ts as UTCTimestamp)
  }
  return (ts) => {
    const direct = scale.timeToCoordinate(ts as UTCTimestamp)
    if (direct != null) return direct
    const first = sorted[0].ts
    const last = sorted[sorted.length - 1].ts
    if (ts <= first || ts >= last) return null
    let lo = 0
    let hi = sorted.length - 1
    while (lo < hi - 1) {
      const mid = (lo + hi) >> 1
      if (sorted[mid].ts <= ts) lo = mid
      else hi = mid
    }
    const xa = scale.timeToCoordinate(sorted[lo].ts as UTCTimestamp)
    const xb = scale.timeToCoordinate(sorted[hi].ts as UTCTimestamp)
    if (xa == null || xb == null) return null
    const t = (ts - sorted[lo].ts) / (sorted[hi].ts - sorted[lo].ts)
    return xa + (xb - xa) * t
  }
}

// Lightweight Charts requires strictly increasing, deduplicated timestamps.
function dedupeAndSort(history: PricePoint[]): PricePoint[] {
  const sorted = [...history].sort((a, b) => a.ts - b.ts)
  const out: PricePoint[] = []
  let lastTs = -Infinity
  for (const p of sorted) {
    if (p.ts === lastTs) continue
    out.push(p)
    lastTs = p.ts
  }
  return out
}

function buildMarkers(data: AssetDoc, overlay: 'ai' | 'actual' | 'none'): SeriesMarker<Time>[] {
  const series = data.history
  if (!series.length) return []
  const trades = data.trades ?? []

  const out: SeriesMarker<Time>[] = []
  for (const m of dedupeTrades(trades)) {
    out.push(tradeMarker(m.bet, m.ts, m.count))
  }

  if (overlay !== 'none') {
    const strat = STRATEGIES.find((s) => s.key === overlay)
    if (strat) {
      const actualBet = makeActualBetLookup(trades)
      const stride = Math.max(1, Math.floor(series.length / 60))
      for (let i = 0; i < series.length; i += stride) {
        const bet = strat.decide(i, series, { actualBet })
        if (!bet) continue
        out.push(overlayMarker(bet, series[i].ts, strat.color))
      }
    }
  }

  // Lightweight Charts requires markers in ascending time order.
  out.sort((a, b) => (a.time as number) - (b.time as number))
  return out
}

function tradeMarker(bet: Bet, ts: number, count: number): SeriesMarker<Time> {
  return {
    time: ts as UTCTimestamp,
    position: bet === 'UP' ? 'belowBar' : 'aboveBar',
    color: bet === 'UP' ? '#34c759' : '#ff3b30',
    shape: 'circle',
    size: count > 1 ? 1.4 : 1,
    text: count > 1 ? `×${count}` : undefined,
  }
}

// Many bots have multiple bets at identical joined_at (the Python builder
// stubs joined_at for active batches). Stacking 20 arrows on one tick is
// noise — collapse them into a single annotated marker.
interface TradeGroup { ts: number; bet: Bet; count: number }
function dedupeTrades(trades: Trade[]): TradeGroup[] {
  const groups = new Map<string, TradeGroup>()
  for (const t of trades) {
    const key = `${t.joined_at}:${t.bet}`
    const g = groups.get(key)
    if (g) g.count++
    else groups.set(key, { ts: t.joined_at, bet: t.bet, count: 1 })
  }
  return Array.from(groups.values())
}

function overlayMarker(bet: Bet, ts: number, color: string): SeriesMarker<Time> {
  return {
    time: ts as UTCTimestamp,
    position: bet === 'UP' ? 'belowBar' : 'aboveBar',
    color,
    shape: 'circle',
    size: 0.5,
  }
}

// priceAt is exported only so other panels can reuse it in the future.
export { priceAt }
