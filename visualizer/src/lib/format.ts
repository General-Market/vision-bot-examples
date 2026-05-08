export function fmtNum(n: number | null | undefined): string {
  if (n == null || !isFinite(n)) return '—'
  const a = Math.abs(n)
  if (a >= 1000) return n.toFixed(0)
  if (a >= 1) return n.toFixed(3)
  return n.toPrecision(4)
}

export function fmtUsd(n: number | null | undefined, decimals = 2): string {
  if (n == null || !isFinite(n)) return '—'
  const sign = n > 0 ? '+' : ''
  return `${sign}$${n.toFixed(decimals)}`
}

export function fmtSignedInt(n: number | null | undefined): string {
  if (n == null || !isFinite(n)) return '—'
  const sign = n > 0 ? '+' : ''
  return `${sign}${n.toFixed(0)}`
}

export function shortAddr(addr: string | undefined | null): string {
  if (!addr) return '…'
  return `${addr.slice(0, 10)}…${addr.slice(-4)}`
}

export function fmtTimestamp(ms: number): string {
  const d = new Date(ms)
  const month = d.toLocaleString('en-US', { month: 'short' })
  const day = d.getDate()
  const hh = String(d.getHours()).padStart(2, '0')
  const mm = String(d.getMinutes()).padStart(2, '0')
  return `${month} ${day} ${hh}:${mm}`
}
