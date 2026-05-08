import type { ReactNode } from 'react'

/*
 * AppShell — light page background; everything inside a centered white
 * card with rounded corners and a single soft shadow. On phones the
 * card goes full-bleed (no margin, no radius) so screen real estate
 * isn't wasted on chrome.
 */
export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen w-full bg-bg sm:p-4 lg:p-6">
      <div
        className="mx-auto h-screen w-full max-w-[1400px] overflow-hidden bg-panel sm:h-[calc(100vh-2rem)] lg:h-[calc(100vh-3rem)]"
        style={{
          borderRadius: 'var(--card-radius, 0)',
          boxShadow: 'var(--card-shadow, none)',
        }}
      >
        <style>{`
          @media (min-width: 640px) {
            :root { --card-radius: var(--radius-xl); --card-shadow: var(--shadow-card); }
          }
        `}</style>
        {children}
      </div>
    </div>
  )
}
