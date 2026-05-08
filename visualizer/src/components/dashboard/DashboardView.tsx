import { useMemo } from 'react'
import type { IndexItem } from '@/lib/types'
import { HeroCard } from './HeroCard'
import { AssetCard } from './AssetCard'

interface Props {
  items: IndexItem[]
  onOpen: (item: IndexItem) => void
  loading: boolean
}

export function DashboardView({ items, onOpen, loading }: Props) {
  const { featured, sideRail, topBacktests } = useMemo(() => pickFeatured(items), [items])

  if (loading && items.length === 0) {
    return (
      <main className="row-start-2 col-start-1 grid min-h-0 place-items-center bg-panel md:col-start-2">
        <div className="text-muted" style={{ fontSize: 14 }}>loading…</div>
      </main>
    )
  }

  if (!featured) {
    return (
      <main className="row-start-2 col-start-1 grid min-h-0 place-items-center bg-panel md:col-start-2">
        <div className="px-6 text-center">
          <div className="font-display font-semibold text-text" style={{ fontSize: 28, letterSpacing: '-0.022em' }}>
            No assets yet.
          </div>
          <div className="mt-2 text-muted" style={{ fontSize: 14 }}>
            Run the build to populate the index.
          </div>
        </div>
      </main>
    )
  }

  return (
    <main className="row-start-2 col-start-1 min-h-0 overflow-y-auto bg-panel md:col-start-2">
      <div className="mx-auto max-w-[1200px] px-4 py-6 sm:px-8 sm:py-8">
        <h1
          className="font-display font-semibold text-text"
          style={{ fontSize: 28, letterSpacing: '-0.022em', lineHeight: 1.07 }}
        >
          Home
        </h1>

        <div className="mt-5 sm:mt-6">
          <HeroCard featured={featured} side={sideRail} onOpen={onOpen} />
        </div>

        <section className="mt-8 sm:mt-12">
          <SectionHeader title="Top backtests" onSeeAll={() => onOpen(topBacktests[0])} />
          <div className="mt-4 grid grid-cols-2 gap-4 sm:gap-6 lg:grid-cols-4">
            {topBacktests.map((it) => (
              <AssetCard key={it.asset_id} item={it} onClick={() => onOpen(it)} />
            ))}
          </div>
        </section>

        <section className="mt-8 mb-4 sm:mt-12">
          <SectionHeader title="Recently active" />
          <div className="mt-4 grid grid-cols-2 gap-4 sm:gap-6 lg:grid-cols-4">
            {recentActive(items).map((it) => (
              <AssetCard key={it.asset_id} item={it} onClick={() => onOpen(it)} />
            ))}
          </div>
        </section>
      </div>
    </main>
  )
}

function SectionHeader({ title, onSeeAll }: { title: string; onSeeAll?: () => void }) {
  return (
    <div className="flex items-baseline justify-between">
      <h2
        className="font-display font-semibold text-text"
        style={{ fontSize: 22, letterSpacing: '-0.022em' }}
      >
        {title}
      </h2>
      {onSeeAll && (
        <button
          type="button"
          onClick={onSeeAll}
          className="cursor-pointer border border-line bg-panel text-text transition hover:bg-[rgba(0,0,0,0.04)]"
          style={{
            borderRadius: 'var(--radius-pill)',
            padding: '6px 14px',
            fontSize: 12,
            fontWeight: 500,
          }}
        >
          See All ›
        </button>
      )}
    </div>
  )
}

function pickFeatured(items: IndexItem[]): {
  featured: IndexItem | null
  sideRail: IndexItem[]
  topBacktests: IndexItem[]
} {
  if (items.length === 0) return { featured: null, sideRail: [], topBacktests: [] }
  const settled = items.filter((i) => i.settled_pnl_usdc != null)
  const byPnl = [...settled].sort(
    (a, b) => (b.settled_pnl_usdc ?? 0) - (a.settled_pnl_usdc ?? 0),
  )
  const byTrades = [...items].sort((a, b) => b.trade_count - a.trade_count)
  const featured = byPnl[0] ?? byTrades[0]
  const sideRail = byTrades.filter((i) => i.asset_id !== featured.asset_id).slice(0, 4)
  const topBacktests = byPnl.length >= 4 ? byPnl.slice(0, 4) : byTrades.slice(0, 4)
  return { featured, sideRail, topBacktests }
}

function recentActive(items: IndexItem[]): IndexItem[] {
  return [...items].sort((a, b) => b.last_seen_at - a.last_seen_at).slice(0, 4)
}
