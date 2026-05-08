import { useMemo, useState } from 'react'
import { clsx } from 'clsx'
import { useAsset } from '@/lib/queries'
import type { IndexItem } from '@/lib/types'
import { aiBacktest, actualPerf, sliceByRange, RANGES, RANGE_LABEL, type Range } from '@/lib/backtest'
import { sourceHue } from '@/lib/view'
import { PriceChart } from './PriceChart'
import { PnLChart } from './PnLChart'

type Tab = 'price' | 'backtest'
type OverlayKey = 'ai' | 'actual' | 'none'

interface Props {
  item: IndexItem | null
  onBack?: () => void
}

export function AssetView({ item, onBack }: Props) {
  const { data, error, isFetching } = useAsset(item?.file ?? null)
  const [range, setRange] = useState<Range>('all')
  const [tab, setTab] = useState<Tab>('price')
  const [overlay, setOverlay] = useState<OverlayKey>('ai')
  const [showCloses, setShowCloses] = useState(true)
  const [showSettlements, setShowSettlements] = useState(false)

  const sliced = useMemo(() => (data ? sliceByRange(data, range) : null), [data, range])

  const priceHero = useMemo(() => {
    if (!sliced || sliced.history.length < 2) return null
    const first = sliced.history[0].price
    const last = sliced.history[sliced.history.length - 1].price
    const delta = last - first
    const pct = first === 0 ? 0 : (delta / first) * 100
    return { last, delta, pct }
  }, [sliced])

  const aiHero = useMemo(() => (sliced ? aiBacktest(sliced.history) : null), [sliced])
  const actualHero = useMemo(
    () => (sliced ? actualPerf(sliced.trades ?? []) : null),
    [sliced],
  )

  const empty = !item

  return (
    <main className="row-start-2 col-start-1 grid min-h-0 min-w-0 grid-rows-[auto_auto_1fr_auto] bg-panel md:col-start-2">
      <Header
        title={empty ? 'Pick an asset.' : (data?.asset_name ?? item.asset_name)}
        sourceName={data?.source_name ?? item?.source_name}
        onBack={onBack}
        range={range}
        onRangeChange={setRange}
      />

      <HeroStrip
        tab={tab}
        onTabChange={setTab}
        price={priceHero}
        ai={aiHero}
        actual={actualHero}
        loading={isFetching && !data}
      />

      <section className="relative flex min-h-0 min-w-0 flex-col px-4 sm:px-6">
        <div className="relative min-h-0 flex-1">
          {empty ? (
            <Empty title="Nothing loaded." hint="Pick an asset on the left." />
          ) : error ? (
            <Empty title={`Couldn't load: ${error.message}`} tone="error" hint="Reload, or rebuild data." />
          ) : !sliced ? (
            <Empty title={`Loading ${item.asset_name}…`} />
          ) : sliced.history.length < 2 ? (
            <Empty title="Not enough price history yet." />
          ) : tab === 'price' ? (
            <PriceChart
              data={sliced}
              overlay={overlay}
              showCloses={showCloses}
              showSettlements={showSettlements}
            />
          ) : (
            <PnLChart data={sliced} stratOn={{ ai: true, actual: true }} />
          )}
        </div>
      </section>

      <Footer
        tab={tab}
        overlay={overlay}
        onOverlayChange={setOverlay}
        showCloses={showCloses}
        onCloseToggle={() => setShowCloses((v) => !v)}
        showSettlements={showSettlements}
        onSettlementToggle={() => setShowSettlements((v) => !v)}
        ai={aiHero}
        actual={actualHero}
      />
    </main>
  )
}

/* ───────── header: back · title · range pills ───────── */

function Header({
  title,
  sourceName,
  onBack,
  range,
  onRangeChange,
}: {
  title: string
  sourceName: string | undefined
  onBack?: () => void
  range: Range
  onRangeChange: (r: Range) => void
}) {
  return (
    <div className="grid grid-cols-[auto_1fr_auto] items-center gap-3 border-b border-line px-4 py-3 sm:gap-4 sm:px-6 sm:py-4">
      {onBack ? (
        <button
          type="button"
          onClick={onBack}
          className="grid h-9 w-9 cursor-pointer place-items-center rounded-full text-muted transition hover:bg-[rgba(0,0,0,0.05)] hover:text-text"
          title="Back"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
            <path d="m15 6-6 6 6 6" />
          </svg>
        </button>
      ) : (
        <span />
      )}
      <div className="flex min-w-0 items-center gap-2">
        <h1
          className="truncate font-display font-semibold text-text"
          style={{ fontSize: 22, letterSpacing: '-0.022em', lineHeight: 1.1 }}
        >
          {title}
        </h1>
        {sourceName && <SourcePill name={sourceName} />}
      </div>
      <RangeSwitcher range={range} onChange={onRangeChange} />
    </div>
  )
}

function SourcePill({ name }: { name: string }) {
  const hue = sourceHue(name)
  return (
    <span
      className="hidden shrink-0 items-center gap-1.5 rounded-full px-2.5 py-1 text-muted sm:inline-flex"
      style={{ fontSize: 11, background: 'rgba(0,0,0,0.05)' }}
      title={name}
    >
      <span
        className="h-2.5 w-2.5 rounded-full"
        style={{
          background: `linear-gradient(135deg, hsl(${hue} 70% 55%), hsl(${(hue + 40) % 360} 70% 45%))`,
        }}
      />
      <span className="max-w-[120px] truncate">{name}</span>
    </span>
  )
}

function RangeSwitcher({
  range,
  onChange,
}: {
  range: Range
  onChange: (r: Range) => void
}) {
  return (
    <div
      className="inline-flex items-center gap-0.5 p-0.5"
      style={{
        background: 'rgba(0,0,0,0.05)',
        borderRadius: 'var(--radius-pill)',
      }}
    >
      {RANGES.map((r) => (
        <button
          key={r}
          type="button"
          onClick={() => onChange(r)}
          className={clsx(
            'cursor-pointer transition',
            range === r ? 'bg-panel text-text shadow-sm' : 'text-muted hover:text-text',
          )}
          style={{
            padding: '4px 10px',
            borderRadius: 'var(--radius-pill)',
            fontSize: 12,
            fontWeight: 500,
            letterSpacing: '-0.01em',
          }}
        >
          {RANGE_LABEL[r]}
        </button>
      ))}
    </div>
  )
}

/* ───────── hero: big number + delta + tab switcher ───────── */

function HeroStrip({
  tab,
  onTabChange,
  price,
  ai,
  actual,
  loading,
}: {
  tab: Tab
  onTabChange: (t: Tab) => void
  price: { last: number; delta: number; pct: number } | null
  ai: ReturnType<typeof aiBacktest> | null
  actual: ReturnType<typeof actualPerf> | null
  loading: boolean
}) {
  const onPriceTab = tab === 'price'
  const heroNumber = onPriceTab ? formatPrice(price?.last) : formatSignedInt(ai?.final ?? null)
  const delta = onPriceTab
    ? formatPriceDelta(price?.delta ?? null, price?.pct ?? null)
    : formatBacktestDelta(ai)
  const positive = onPriceTab ? (price?.delta ?? 0) >= 0 : (ai?.final ?? 0) >= 0

  const subline = onPriceTab
    ? actual && actual.wins + actual.losses + actual.open > 0
      ? `${actual.wins + actual.losses + actual.open} bot trades · ${actual.wins}W / ${actual.losses}L`
      : null
    : ai && ai.ticks > 0
      ? `${ai.ticks} ticks · ${ai.wins}W / ${ai.losses}L`
      : null

  return (
    <div className="flex flex-wrap items-end justify-between gap-4 px-4 py-4 sm:px-6 sm:py-5">
      <div className="min-w-0">
        <div className="text-faint" style={{ fontSize: 11, letterSpacing: '0.06em', textTransform: 'uppercase' }}>
          {onPriceTab ? 'Last price' : 'AI backtest score'}
        </div>
        <div
          className={clsx(
            'mt-0.5 font-display font-semibold tabular-nums text-text',
          )}
          style={{ fontSize: 40, letterSpacing: '-0.022em', lineHeight: 1.05 }}
        >
          {loading ? '…' : heroNumber}
        </div>
        {delta && (
          <div
            className={clsx(
              'mt-1 num font-medium',
              positive ? 'text-up' : 'text-down',
            )}
            style={{ fontSize: 14 }}
          >
            {delta}
          </div>
        )}
        {subline && (
          <div className="mt-0.5 text-muted" style={{ fontSize: 12 }}>
            {subline}
          </div>
        )}
      </div>

      <TabSwitcher tab={tab} onChange={onTabChange} />
    </div>
  )
}

function TabSwitcher({ tab, onChange }: { tab: Tab; onChange: (t: Tab) => void }) {
  return (
    <div
      className="inline-flex items-center gap-0.5 p-0.5"
      style={{
        background: 'rgba(0,0,0,0.05)',
        borderRadius: 'var(--radius-pill)',
      }}
    >
      {(['price', 'backtest'] as const).map((t) => (
        <button
          key={t}
          type="button"
          onClick={() => onChange(t)}
          className={clsx(
            'cursor-pointer transition',
            tab === t ? 'bg-panel text-text shadow-sm' : 'text-muted hover:text-text',
          )}
          style={{
            padding: '6px 14px',
            borderRadius: 'var(--radius-pill)',
            fontSize: 13,
            fontWeight: 500,
            letterSpacing: '-0.01em',
          }}
        >
          {t === 'price' ? 'Price' : 'Backtest'}
        </button>
      ))}
    </div>
  )
}

/* ───────── footer: contextual controls ───────── */

function Footer({
  tab,
  overlay,
  onOverlayChange,
  showCloses,
  onCloseToggle,
  showSettlements,
  onSettlementToggle,
  ai,
  actual,
}: {
  tab: Tab
  overlay: OverlayKey
  onOverlayChange: (o: OverlayKey) => void
  showCloses: boolean
  onCloseToggle: () => void
  showSettlements: boolean
  onSettlementToggle: () => void
  ai: ReturnType<typeof aiBacktest> | null
  actual: ReturnType<typeof actualPerf> | null
}) {
  if (tab === 'price') {
    return (
      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line px-4 py-3 sm:px-6">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2" style={{ fontSize: 12 }}>
          <div className="flex items-center gap-1.5">
            <span className="text-muted pr-1">Overlay</span>
            {(['ai', 'actual', 'none'] as const).map((k) => (
              <button
                key={k}
                type="button"
                onClick={() => onOverlayChange(k)}
                className={clsx(
                  'cursor-pointer transition',
                  overlay === k
                    ? 'bg-text text-white'
                    : 'border border-line text-muted hover:text-text hover:bg-[rgba(0,0,0,0.04)]',
                )}
                style={{
                  borderRadius: 'var(--radius-pill)',
                  padding: overlay === k ? '5px 12px' : '4px 11px',
                }}
              >
                {k === 'ai' ? 'AI' : k === 'actual' ? "bot's actual" : 'none'}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-muted pr-1">Lines</span>
            <LineToggle
              on={showCloses}
              onClick={onCloseToggle}
              swatch={<DashedSwatch color="rgba(110,110,115,0.7)" />}
              label="Closes"
            />
            <LineToggle
              on={showSettlements}
              onClick={onSettlementToggle}
              swatch={<SolidSwatch color="rgba(255,59,48,0.75)" />}
              label="Settlements"
            />
          </div>
        </div>
        {actual && actual.wins + actual.losses > 0 && (
          <div className="text-muted" style={{ fontSize: 12 }}>
            <span className="text-up num font-medium">{actual.wins}W</span>
            <span className="px-1 text-faint">/</span>
            <span className="text-down num font-medium">{actual.losses}L</span>
            {actual.open > 0 && (
              <>
                <span className="px-1 text-faint">·</span>
                <span className="num font-medium text-warn">{actual.open}</span> open
              </>
            )}
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line px-4 py-3 sm:px-6">
      <div className="flex items-center gap-3" style={{ fontSize: 12 }}>
        <Legend label="AI" color="#ff9500" />
        <Legend label="bot's actual" color="#1d1d1f" />
      </div>
      {ai && (
        <div className="text-muted" style={{ fontSize: 12 }}>
          {ai.ticks} ticks ·{' '}
          <span className="text-up num font-medium">{ai.wins}W</span>
          <span className="px-1 text-faint">/</span>
          <span className="text-down num font-medium">{ai.losses}L</span>
        </div>
      )}
    </div>
  )
}

function Legend({ label, color }: { label: string; color: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-muted">
      <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: color }} />
      {label}
    </span>
  )
}

function LineToggle({
  on,
  onClick,
  swatch,
  label,
}: {
  on: boolean
  onClick: () => void
  swatch: React.ReactNode
  label: string
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={clsx(
        'inline-flex cursor-pointer items-center gap-1.5 transition',
        on
          ? 'border border-text bg-panel text-text'
          : 'border border-line text-muted hover:text-text hover:bg-[rgba(0,0,0,0.04)]',
      )}
      style={{
        borderRadius: 'var(--radius-pill)',
        padding: '4px 11px',
      }}
    >
      {swatch}
      {label}
    </button>
  )
}

function DashedSwatch({ color }: { color: string }) {
  return (
    <span
      className="inline-block"
      style={{
        width: 14,
        height: 0,
        borderBottom: `1.5px dashed ${color}`,
      }}
    />
  )
}

function SolidSwatch({ color }: { color: string }) {
  return (
    <span
      className="inline-block"
      style={{
        width: 14,
        height: 0,
        borderBottom: `1.5px solid ${color}`,
      }}
    />
  )
}

/* ───────── empty / loading ───────── */

function Empty({
  title,
  hint,
  tone,
}: {
  title: string
  hint?: string
  tone?: 'error'
}) {
  return (
    <div className="absolute inset-0 grid place-items-center bg-panel pointer-events-none">
      <div className="text-center">
        <div className={tone === 'error' ? 'text-down' : 'text-text'} style={{ fontSize: 15 }}>
          {title}
        </div>
        {hint && (
          <div className="mt-1.5 text-muted" style={{ fontSize: 12 }}>
            {hint}
          </div>
        )}
      </div>
    </div>
  )
}

/* ───────── number formatting ───────── */

function formatPrice(n: number | undefined): string {
  if (n == null || !isFinite(n)) return '—'
  if (Math.abs(n) >= 1000) return n.toFixed(0)
  if (Math.abs(n) >= 1) return n.toFixed(2)
  return n.toPrecision(4)
}

function formatPriceDelta(delta: number | null, pct: number | null): string | null {
  if (delta == null || pct == null) return null
  const sign = delta >= 0 ? '+' : ''
  const a = Math.abs(delta)
  const dStr = a >= 100 ? delta.toFixed(0) : a >= 1 ? delta.toFixed(2) : delta.toPrecision(3)
  return `${sign}${dStr} (${sign}${pct.toFixed(2)}%)`
}

function formatSignedInt(n: number | null): string {
  if (n == null) return '—'
  const sign = n > 0 ? '+' : ''
  return `${sign}${n}`
}

function formatBacktestDelta(ai: ReturnType<typeof aiBacktest> | null): string | null {
  if (!ai || ai.ticks === 0) return null
  const winRate = (ai.wins / ai.ticks) * 100
  return `${winRate.toFixed(1)}% win rate`
}
