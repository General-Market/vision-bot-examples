import { useMemo, useRef } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'
import { clsx } from 'clsx'
import type { Bet, IndexItem } from '@/lib/types'
import { applyFilter, sourceHue, type FilterKey } from '@/lib/view'

const ROW_HEIGHT = 64

interface Props {
  items: IndexItem[]
  selected: IndexItem | null
  onSelect: (item: IndexItem) => void
  query: string
  filter: FilterKey
  loading: boolean
}

const FILTER_LABEL: Record<FilterKey, string> = {
  all: 'Browse',
  up: 'UP only',
  down: 'DOWN only',
  settled: 'Settled',
  open: 'Open',
}

export function AssetList({ items, selected, onSelect, query, filter, loading }: Props) {
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    let arr = applyFilter(items, filter)
    if (q) {
      arr = arr.filter((it) =>
        `${it.asset_id} ${it.asset_name} ${it.source_name}`.toLowerCase().includes(q),
      )
    }
    return arr
  }, [items, query, filter])

  const scrollerRef = useRef<HTMLDivElement | null>(null)
  const virtualizer = useVirtualizer({
    count: filtered.length,
    getScrollElement: () => scrollerRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 8,
  })

  return (
    <main className="row-start-2 col-start-1 grid min-h-0 min-w-0 grid-rows-[auto_1fr] bg-panel md:col-start-2">
      <header className="flex flex-wrap items-baseline justify-between gap-2 border-b border-line px-4 pt-5 pb-3 sm:px-8 sm:pt-6 sm:pb-4">
        <h1
          className="font-display font-semibold text-text"
          style={{ fontSize: 24, letterSpacing: '-0.022em', lineHeight: 1.1 }}
        >
          {FILTER_LABEL[filter]}
        </h1>
        <span className="text-muted" style={{ fontSize: 13 }}>
          {filtered.length} {filtered.length === 1 ? 'asset' : 'assets'}
          {filtered.length !== items.length ? ` · ${items.length} total` : ''}
        </span>
      </header>

      <div ref={scrollerRef} className="relative overflow-y-auto px-2 py-2 sm:px-4">
        {loading && (
          <div className="px-4 py-3 text-muted" style={{ fontSize: 13 }}>
            loading…
          </div>
        )}
        {!loading && filtered.length === 0 && (
          <div className="grid place-items-center px-4 py-16 text-center">
            <div>
              <div className="text-text" style={{ fontSize: 17 }}>
                Nothing matches.
              </div>
              <div className="mt-1 text-muted" style={{ fontSize: 13 }}>
                Try a different filter or clear the search.
              </div>
            </div>
          </div>
        )}
        <div style={{ height: virtualizer.getTotalSize(), position: 'relative' }}>
          {virtualizer.getVirtualItems().map((row) => {
            const it = filtered[row.index]
            return (
              <Row
                key={it.asset_id}
                item={it}
                active={selected?.asset_id === it.asset_id}
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  height: ROW_HEIGHT,
                  transform: `translateY(${row.start}px)`,
                }}
                onClick={() => onSelect(it)}
              />
            )
          })}
        </div>
      </div>
    </main>
  )
}

function Row({
  item,
  active,
  style,
  onClick,
}: {
  item: IndexItem
  active: boolean
  style: React.CSSProperties
  onClick: () => void
}) {
  const last = item.last_bet
  const pnl = item.settled_pnl_usdc
  const pnlText = pnl == null ? '' : (pnl > 0 ? '+' : '') + pnl.toFixed(2)
  const hue = sourceHue(item.source_name)
  return (
    <div
      style={style}
      onClick={onClick}
      className={clsx(
        'grid cursor-pointer grid-cols-[40px_1fr_auto_auto] items-center gap-4 rounded-xl px-3 transition',
        'hover:bg-[rgba(0,0,0,0.03)]',
        active && 'bg-[rgba(0,113,227,0.08)]',
      )}
    >
      <div
        className="grid h-9 w-9 place-items-center rounded-full font-medium text-white"
        style={{
          fontSize: 12,
          background: `linear-gradient(135deg, hsl(${hue} 70% 55%), hsl(${(hue + 40) % 360} 70% 45%))`,
        }}
      >
        {item.source_name.slice(0, 1).toUpperCase()}
      </div>
      <div className="overflow-hidden">
        <div
          className="truncate font-medium text-text"
          style={{ fontSize: 15, letterSpacing: '-0.016em' }}
        >
          {item.asset_name}
        </div>
        <div className="truncate text-muted" style={{ fontSize: 12 }}>
          {item.source_name} · {item.trade_count} {item.trade_count === 1 ? 'trade' : 'trades'}
        </div>
      </div>
      <BetPill bet={last} />
      <div
        className={clsx(
          'num min-w-[64px] text-right',
          pnl == null ? 'text-faint' : pnl > 0 ? 'text-up' : pnl < 0 ? 'text-down' : 'text-faint',
        )}
        style={{ fontSize: 13 }}
      >
        {pnlText || '—'}
      </div>
    </div>
  )
}

function BetPill({ bet }: { bet: Bet | null }) {
  if (!bet) {
    return (
      <span className="text-faint" style={{ fontSize: 11 }}>
        —
      </span>
    )
  }
  const isUp = bet === 'UP'
  return (
    <span
      className="inline-block rounded-full px-2 py-0.5 font-semibold"
      style={{
        fontSize: 10,
        letterSpacing: '0.04em',
        background: isUp ? 'rgba(52,199,89,0.15)' : 'rgba(255,59,48,0.15)',
        color: isUp ? '#0a8035' : '#c41e15',
      }}
    >
      {bet}
    </span>
  )
}
