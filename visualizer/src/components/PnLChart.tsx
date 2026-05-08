import { useEffect, useMemo, useRef } from 'react'
import {
  ColorType,
  CrosshairMode,
  LineSeries,
  createChart,
  type IChartApi,
  type ISeriesApi,
  type UTCTimestamp,
} from 'lightweight-charts'
import type { AssetDoc, PricePoint } from '@/lib/types'
import { STRATEGIES, makeActualBetLookup } from '@/lib/strategies'

type StratKey = 'ai' | 'actual'

interface Props {
  data: AssetDoc
  stratOn: Record<StratKey, boolean>
}

export function PnLChart({ data, stratOn }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesRef = useRef<Map<StratKey, ISeriesApi<'Line'>>>(new Map())

  const series = useMemo(() => {
    const sortedHistory = dedupeAndSort(data.history)
    if (sortedHistory.length < 11) return null
    const actualBet = makeActualBetLookup(data.trades ?? [])
    const out: Record<StratKey, { time: UTCTimestamp; value: number }[]> = {
      ai: [],
      actual: [],
    }
    const totals: Record<StratKey, number> = { ai: 0, actual: 0 }

    for (const s of STRATEGIES) {
      let cum = 0
      const data: { time: UTCTimestamp; value: number }[] = [
        { time: sortedHistory[0].ts as UTCTimestamp, value: 0 },
      ]
      for (let i = 0; i + 1 < sortedHistory.length; i++) {
        const bet = s.decide(i, sortedHistory, { actualBet })
        if (bet) {
          const moveUp = sortedHistory[i + 1].price > sortedHistory[i].price
          const moveDown = sortedHistory[i + 1].price < sortedHistory[i].price
          if (moveUp || moveDown) {
            cum += (bet === 'UP' && moveUp) || (bet === 'DOWN' && moveDown) ? 1 : -1
          }
        }
        data.push({ time: sortedHistory[i + 1].ts as UTCTimestamp, value: cum })
      }
      out[s.key as StratKey] = data
      totals[s.key as StratKey] = cum
    }
    return { lines: out, totals }
  }, [data])

  // Mount chart once.
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
    chartRef.current = chart

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
      seriesRef.current.clear()
    }
  }, [])

  // Update lines whenever data or toggles change.
  useEffect(() => {
    const chart = chartRef.current
    if (!chart) return

    const map = seriesRef.current
    for (const s of STRATEGIES) {
      const key = s.key as StratKey
      const enabled = stratOn[key] && series && series.lines[key].length > 0
      if (!enabled) {
        const existing = map.get(key)
        if (existing) {
          chart.removeSeries(existing)
          map.delete(key)
        }
        continue
      }
      let line = map.get(key)
      if (!line) {
        line = chart.addSeries(LineSeries, {
          color: s.color,
          lineWidth: 2,
          priceLineVisible: false,
          lastValueVisible: false,
        })
        map.set(key, line)
      }
      line.setData(series!.lines[key])
    }
    chart.timeScale().fitContent()
  }, [series, stratOn])

  return <div ref={containerRef} className="absolute inset-0" />
}

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
