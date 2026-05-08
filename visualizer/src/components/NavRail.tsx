import { useMemo, useState } from 'react'
import { clsx } from 'clsx'
import type { IndexItem } from '@/lib/types'
import { type FilterKey, type View, sourceHue } from '@/lib/view'

interface Props {
  view: View
  onChange: (v: View) => void
  items: IndexItem[]
  sources: string[]
  mobileOpen: boolean
  onClose: () => void
}

const VISIBLE_SOURCES = 6

export function NavRail({ view, onChange, items, sources, mobileOpen, onClose }: Props) {
  const [showAllSources, setShowAllSources] = useState(false)
  const sourceCounts = useMemo(() => {
    const m = new Map<string, number>()
    for (const it of items) m.set(it.source_name, (m.get(it.source_name) ?? 0) + 1)
    return m
  }, [items])

  const sortedSources = useMemo(
    () => [...sources].sort((a, b) => (sourceCounts.get(b) ?? 0) - (sourceCounts.get(a) ?? 0)),
    [sources, sourceCounts],
  )

  const visibleSources = showAllSources ? sortedSources : sortedSources.slice(0, VISIBLE_SOURCES)

  return (
    <>
      {mobileOpen && (
        <div
          className="fixed inset-0 z-20 bg-black/30 md:hidden"
          onClick={onClose}
          aria-hidden
        />
      )}
      <aside
        className={clsx(
          'flex min-h-0 flex-col gap-1 overflow-y-auto border-r border-line bg-panel px-3 py-3 transition-transform',
          'fixed inset-y-0 left-0 z-30 w-[260px]',
          'md:relative md:row-start-2 md:col-start-1 md:w-auto md:translate-x-0 md:transition-none',
          mobileOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0',
        )}
      >
      <NavSection>
        <NavRow
          icon={<HomeIcon />}
          label="Home"
          active={view.kind === 'home'}
          onClick={() => onChange({ kind: 'home' })}
        />
        <NavRow
          icon={<CompassIcon />}
          label="Browse"
          active={view.kind === 'browse' && view.filter === 'all'}
          onClick={() => onChange({ kind: 'browse', filter: 'all' })}
        />
        <NavRow
          icon={<BriefcaseIcon />}
          label="Strategies"
          active={view.kind === 'strategies'}
          onClick={() => onChange({ kind: 'strategies' })}
          dim
        />
      </NavSection>

      <Divider />

      <NavSection>
        <FilterRow
          icon={<TrendingUpIcon />}
          label="UP only"
          filter="up"
          view={view}
          onChange={onChange}
        />
        <FilterRow
          icon={<TrendingDownIcon />}
          label="DOWN only"
          filter="down"
          view={view}
          onChange={onChange}
        />
        <FilterRow
          icon={<CheckIcon />}
          label="Settled"
          filter="settled"
          view={view}
          onChange={onChange}
        />
        <FilterRow
          icon={<ClockIcon />}
          label="Open"
          filter="open"
          view={view}
          onChange={onChange}
        />
      </NavSection>

      <Divider />

      <div className="px-2 pt-1 pb-1.5">
        <span
          className="font-medium text-faint"
          style={{ fontSize: 11, letterSpacing: '0.04em', textTransform: 'uppercase' }}
        >
          Sources
        </span>
      </div>
      <NavSection>
        {visibleSources.map((src) => (
          <SourceRow key={src} name={src} count={sourceCounts.get(src) ?? 0} />
        ))}
        {sortedSources.length > VISIBLE_SOURCES && (
          <NavRow
            icon={<ChevronIcon open={showAllSources} />}
            label={showAllSources ? 'Show Less' : 'Show More'}
            onClick={() => setShowAllSources((v) => !v)}
            dim
          />
        )}
      </NavSection>
      </aside>
    </>
  )
}

function NavSection({ children }: { children: React.ReactNode }) {
  return <div className="flex flex-col gap-0.5">{children}</div>
}

function Divider() {
  return <div className="my-2 h-px bg-line" />
}

function NavRow({
  icon,
  label,
  active = false,
  dim = false,
  onClick,
}: {
  icon: React.ReactNode
  label: string
  active?: boolean
  dim?: boolean
  onClick?: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={clsx(
        'group flex w-full cursor-pointer items-center gap-3 rounded-lg px-2.5 py-2 text-left transition',
        active && 'bg-[rgba(0,0,0,0.06)] text-text font-medium',
        !active && !dim && 'text-text hover:bg-[rgba(0,0,0,0.04)]',
        !active && dim && 'text-muted hover:bg-[rgba(0,0,0,0.04)] hover:text-text',
      )}
      style={{ fontSize: 14, letterSpacing: '-0.01em' }}
    >
      <span className="grid h-5 w-5 place-items-center text-current">{icon}</span>
      <span>{label}</span>
    </button>
  )
}

function FilterRow({
  icon,
  label,
  filter,
  view,
  onChange,
}: {
  icon: React.ReactNode
  label: string
  filter: FilterKey
  view: View
  onChange: (v: View) => void
}) {
  const active = view.kind === 'browse' && view.filter === filter
  return (
    <NavRow
      icon={icon}
      label={label}
      active={active}
      onClick={() => onChange({ kind: 'browse', filter })}
    />
  )
}

function SourceRow({ name, count }: { name: string; count: number }) {
  const hue = sourceHue(name)
  return (
    <div
      className="flex cursor-default items-center gap-3 rounded-lg px-2.5 py-1.5 text-text hover:bg-[rgba(0,0,0,0.04)]"
      title={`${count} ${count === 1 ? 'asset' : 'assets'} from ${name}`}
      style={{ fontSize: 14, letterSpacing: '-0.01em' }}
    >
      <span
        className="h-5 w-5 shrink-0 rounded-full"
        style={{
          background: `linear-gradient(135deg, hsl(${hue} 70% 55%), hsl(${(hue + 40) % 360} 70% 45%))`,
        }}
      />
      <span className="flex-1 truncate">{name}</span>
      <VerifiedCheck />
      <span className="num text-faint" style={{ fontSize: 12 }}>
        {count}
      </span>
    </div>
  )
}

/* ───────── icons (line, 1.6–1.8 stroke, rounded caps) ───────── */

function HomeIcon() {
  return (
    <Svg>
      <path d="M3 11 12 4l9 7" />
      <path d="M5 10v10h14V10" />
    </Svg>
  )
}
function CompassIcon() {
  return (
    <Svg>
      <circle cx="12" cy="12" r="9" />
      <path d="m15 9-2.5 5.5L7 17l2.5-5.5z" />
    </Svg>
  )
}
function BriefcaseIcon() {
  return (
    <Svg>
      <rect x="3" y="7" width="18" height="13" rx="2" />
      <path d="M9 7V5a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2" />
    </Svg>
  )
}
function TrendingUpIcon() {
  return (
    <Svg>
      <path d="m3 17 6-6 4 4 8-8" />
      <path d="M14 7h7v7" />
    </Svg>
  )
}
function TrendingDownIcon() {
  return (
    <Svg>
      <path d="m3 7 6 6 4-4 8 8" />
      <path d="M14 17h7v-7" />
    </Svg>
  )
}
function CheckIcon() {
  return (
    <Svg>
      <path d="M5 12.5 10 17l9-10" />
    </Svg>
  )
}
function ClockIcon() {
  return (
    <Svg>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 7v5l3 2" />
    </Svg>
  )
}
function ChevronIcon({ open }: { open: boolean }) {
  return (
    <Svg style={{ transform: open ? 'rotate(180deg)' : undefined, transition: 'transform 200ms' }}>
      <path d="m6 9 6 6 6-6" />
    </Svg>
  )
}
function VerifiedCheck() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="#0071e3" aria-hidden>
      <path d="M12 2 9.5 4.5 6 4l-1 3.5L2 10l2.5 2.5L4 16l3.5 1L9.5 20 12 18l2.5 2 2-3 3.5-1-.5-3.5L22 10l-3-2.5L18 4l-3.5.5z" />
      <path d="m8 12 3 3 5-6" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" fill="none" />
    </svg>
  )
}
function Svg({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      style={style}
    >
      {children}
    </svg>
  )
}
