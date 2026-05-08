/**
 * General Market mark — black square with horizontal white bars.
 * Source: frontend/public/logo.svg, simplified into a single bar group.
 */
export function GMLogo({ size = 22, className }: { size?: number; className?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 102 102"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-label="General Market"
    >
      <rect width="102" height="102" rx="22" fill="#1d1d1f" />
      <g fill="#ffffff">
        <rect x="15.28" y="48.10" width="14.91" height="5.97" rx="1.47" />
        <rect x="26.62" y="48.10" width="14.91" height="5.97" rx="1.47" />
        <rect x="37.97" y="48.10" width="14.91" height="5.97" rx="1.47" />
        <rect x="49.31" y="48.10" width="14.91" height="5.97" rx="1.47" />
        <rect x="60.65" y="48.12" width="9.24" height="5.95" rx="1.47" />
        <rect x="66.32" y="48.10" width="14.91" height="5.97" rx="1.47" />
        <rect x="77.67" y="48.12" width="9.24" height="5.95" rx="1.47" />
      </g>
    </svg>
  )
}
