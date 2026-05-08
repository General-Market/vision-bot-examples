import { useMemo } from 'react'
import type { PricePoint } from '@/lib/types'

interface Props {
  points: PricePoint[] | undefined
  width?: number
  height?: number
  stroke?: string
  strokeWidth?: number
  fill?: string
  className?: string
}

/**
 * Tiny SVG sparkline. No axes, no labels — just the line and an optional
 * gradient under it. Cheap to render in a card grid.
 */
export function Sparkline({
  points,
  width = 320,
  height = 96,
  stroke = '#0071e3',
  strokeWidth = 1.6,
  fill = 'rgba(0,113,227,0.10)',
  className,
}: Props) {
  const path = useMemo(() => buildPath(points, width, height), [points, width, height])
  if (!path) return null
  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      className={className}
      width="100%"
      height="100%"
      aria-hidden
    >
      {fill && <path d={path.area} fill={fill} />}
      <path d={path.line} fill="none" stroke={stroke} strokeWidth={strokeWidth} strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  )
}

function buildPath(
  points: PricePoint[] | undefined,
  width: number,
  height: number,
): { line: string; area: string } | null {
  if (!points || points.length < 2) return null
  const sorted = [...points].sort((a, b) => a.ts - b.ts)
  const xs = sorted.map((p) => p.ts)
  const ys = sorted.map((p) => p.price)
  const xMin = xs[0]
  const xMax = xs[xs.length - 1]
  const yMin = Math.min(...ys)
  const yMax = Math.max(...ys)
  const xRange = xMax - xMin || 1
  const yRange = yMax - yMin || 1
  const pad = 2

  const project = (i: number): [number, number] => {
    const x = ((xs[i] - xMin) / xRange) * (width - pad * 2) + pad
    const y = height - pad - ((ys[i] - yMin) / yRange) * (height - pad * 2)
    return [x, y]
  }

  const line = sorted
    .map((_, i) => {
      const [x, y] = project(i)
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(2)} ${y.toFixed(2)}`
    })
    .join(' ')

  const area = `${line} L${(width - pad).toFixed(2)} ${(height - pad).toFixed(2)} L${pad.toFixed(2)} ${(height - pad).toFixed(2)} Z`

  return { line, area }
}
