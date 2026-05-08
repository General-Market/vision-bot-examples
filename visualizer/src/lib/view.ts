import type { IndexItem } from './types'

export type FilterKey = 'all' | 'up' | 'down' | 'settled' | 'open'

export type View =
  | { kind: 'home' }
  | { kind: 'browse'; filter: FilterKey }
  | { kind: 'strategies' }

export const HOME: View = { kind: 'home' }
export const BROWSE: View = { kind: 'browse', filter: 'all' }

export function applyFilter(items: IndexItem[], filter: FilterKey): IndexItem[] {
  switch (filter) {
    case 'all':
      return items
    case 'up':
      return items.filter((i) => i.last_bet === 'UP')
    case 'down':
      return items.filter((i) => i.last_bet === 'DOWN')
    case 'settled':
      return items.filter((i) => i.settled_pnl_usdc != null)
    case 'open':
      return items.filter((i) => i.settled_pnl_usdc == null)
  }
}

/** Deterministic hue from a source name → colored dot in the rail. */
export function sourceHue(name: string): number {
  let h = 0
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0
  return h % 360
}
