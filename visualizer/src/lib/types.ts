export type Bet = 'UP' | 'DOWN'

export interface IndexStats {
  active_batches: number
  settled_batches: number
  total_deposited_usdc: number
  total_settled_pnl_usdc: number
  asset_count: number
  trade_count: number
  first_seen_at: number
  last_seen_at: number
  sources: string[]
}

export interface IndexItem {
  asset_id: string
  asset_name: string
  source_name: string
  trade_count: number
  last_bet: Bet | null
  settled_pnl_usdc: number | null
  points: number
  last_seen_at: number
  file: string
}

export interface IndexDoc {
  generated_at: number
  usdc_decimals: number
  player: string
  stats: IndexStats
  items: IndexItem[]
}

export interface PricePoint {
  ts: number
  price: number
}

export interface Trade {
  batch_id: number
  bet: Bet
  joined_at: number
  settled: boolean
  pnl_usdc: number | null
  deposit_usdc: number
  tick_duration: number
  source_name: string
}

export interface AssetDoc {
  asset_id: string
  asset_name: string
  source_name: string
  history: PricePoint[]
  trades: Trade[]
}
