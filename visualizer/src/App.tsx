import { useState } from 'react'
import { AppShell } from './components/AppShell'
import { TopBar } from './components/TopBar'
import { NavRail } from './components/NavRail'
import { AssetList } from './components/AssetList'
import { AssetView } from './components/AssetView'
import { DashboardView } from './components/dashboard/DashboardView'
import { useIndex } from './lib/queries'
import type { IndexItem } from './lib/types'
import { type View } from './lib/view'

export function App() {
  const { data: index, error, isPending } = useIndex()
  const [selected, setSelected] = useState<IndexItem | null>(null)
  const [query, setQuery] = useState('')
  const [view, setView] = useState<View>({ kind: 'home' })
  const [navOpen, setNavOpen] = useState(false)

  const items = index?.items ?? []
  const sources = index?.stats?.sources ?? []

  function changeView(v: View) {
    setView(v)
    setSelected(null)
    setNavOpen(false)
  }

  function selectAsset(item: IndexItem) {
    setSelected(item)
    setNavOpen(false)
  }

  return (
    <AppShell>
      <div className="grid h-full grid-cols-1 grid-rows-[auto_1fr] md:grid-cols-[240px_1fr]">
        <TopBar
          player={index?.player}
          generatedAt={index?.generated_at}
          loading={isPending}
          error={error?.message}
          query={query}
          onQueryChange={setQuery}
          onToggleNav={() => setNavOpen((v) => !v)}
        />
        <NavRail
          view={view}
          onChange={changeView}
          items={items}
          sources={sources}
          mobileOpen={navOpen}
          onClose={() => setNavOpen(false)}
        />
        <MainPanel
          view={view}
          selected={selected}
          items={items}
          query={query}
          loading={isPending}
          onSelect={selectAsset}
          onBack={() => setSelected(null)}
        />
      </div>
    </AppShell>
  )
}

function MainPanel({
  view,
  selected,
  items,
  query,
  loading,
  onSelect,
  onBack,
}: {
  view: View
  selected: IndexItem | null
  items: IndexItem[]
  query: string
  loading: boolean
  onSelect: (item: IndexItem) => void
  onBack: () => void
}) {
  if (selected) return <AssetView item={selected} onBack={onBack} />

  if (view.kind === 'home') {
    return <DashboardView items={items} onOpen={onSelect} loading={loading} />
  }

  if (view.kind === 'strategies') {
    return <StrategiesPlaceholder />
  }

  return (
    <AssetList
      items={items}
      selected={selected}
      onSelect={onSelect}
      query={query}
      filter={view.filter}
      loading={loading}
    />
  )
}

function StrategiesPlaceholder() {
  return (
    <main className="row-start-2 col-start-1 grid min-h-0 place-items-center bg-panel md:col-start-2">
      <div className="text-center px-6">
        <div
          className="font-display font-semibold text-text"
          style={{ fontSize: 28, letterSpacing: '-0.022em' }}
        >
          Strategies
        </div>
        <div className="mt-2 text-muted" style={{ fontSize: 15 }}>
          Strategy comparisons go here.
        </div>
      </div>
    </main>
  )
}
