import { clsx } from 'clsx'
import type { IndexItem } from '@/lib/types'
import { useAsset } from '@/lib/queries'
import { sourceHue } from '@/lib/view'
import { Sparkline } from './Sparkline'

interface Props {
  item: IndexItem
  onClick: () => void
}

export function AssetCard({ item, onClick }: Props) {
  const { data } = useAsset(item.file)
  const hue = sourceHue(item.source_name)
  const pnl = item.settled_pnl_usdc
  const pnlText = pnl == null ? null : (pnl > 0 ? '+' : '') + '$' + pnl.toFixed(2)

  return (
    <button
      type="button"
      onClick={onClick}
      className="group flex cursor-pointer flex-col gap-2.5 text-left transition"
    >
      <div
        className="relative aspect-[16/9] w-full overflow-hidden"
        style={{
          borderRadius: 'var(--radius-md)',
          background: `linear-gradient(135deg, hsl(${hue} 80% 96%), hsl(${(hue + 40) % 360} 80% 92%))`,
        }}
      >
        <div className="absolute inset-0 flex items-end p-3">
          <Sparkline
            points={data?.history}
            width={400}
            height={120}
            stroke={`hsl(${hue} 60% 38%)`}
            fill={`hsla(${hue}, 70%, 50%, 0.18)`}
          />
        </div>
        {item.last_bet && (
          <div
            className="absolute bottom-2 right-2 rounded-full px-2 py-0.5 font-semibold text-white"
            style={{
              fontSize: 10,
              letterSpacing: '0.04em',
              background: 'rgba(0,0,0,0.65)',
            }}
          >
            {item.last_bet}
          </div>
        )}
      </div>
      <div className="flex gap-3">
        <div
          className="grid h-9 w-9 shrink-0 place-items-center rounded-full font-medium text-white"
          style={{
            fontSize: 12,
            background: `linear-gradient(135deg, hsl(${hue} 70% 55%), hsl(${(hue + 40) % 360} 70% 45%))`,
          }}
        >
          {item.source_name.slice(0, 1).toUpperCase()}
        </div>
        <div className="min-w-0 flex-1">
          <div
            className="line-clamp-2 font-medium text-text"
            style={{ fontSize: 14, letterSpacing: '-0.016em', lineHeight: 1.3 }}
          >
            {item.asset_name}
          </div>
          <div className="mt-0.5 flex items-center gap-1 text-muted" style={{ fontSize: 12 }}>
            <span className="truncate">{item.source_name}</span>
            <VerifiedDot />
          </div>
          <div className="mt-0.5 text-faint" style={{ fontSize: 12 }}>
            <span className="num">{item.trade_count}</span>{' '}
            {item.trade_count === 1 ? 'trade' : 'trades'}
            {pnlText && (
              <>
                <span> · </span>
                <span
                  className={clsx(
                    'num font-medium',
                    pnl != null && pnl > 0 && 'text-up',
                    pnl != null && pnl < 0 && 'text-down',
                  )}
                >
                  {pnlText}
                </span>
              </>
            )}
          </div>
        </div>
      </div>
    </button>
  )
}

function VerifiedDot() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="#0071e3" aria-hidden>
      <path d="M12 2 9.5 4.5 6 4l-1 3.5L2 10l2.5 2.5L4 16l3.5 1L9.5 20 12 18l2.5 2 2-3 3.5-1-.5-3.5L22 10l-3-2.5L18 4l-3.5.5z" />
      <path d="m8 12 3 3 5-6" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" fill="none" />
    </svg>
  )
}
