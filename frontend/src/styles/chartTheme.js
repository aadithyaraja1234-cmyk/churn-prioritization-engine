// Shared Recharts styling so every chart on this dark-themed app looks
// consistent, rather than silently falling back to Recharts' own defaults
// (which assume a light page: dark-grey tick text at a bare-minimum
// passing contrast, and a plain WHITE tooltip box - the single most
// visually jarring "chart-library default never updated" case found in
// this audit). CSS custom properties resolve fine inside these inline
// style objects since Recharts renders them into real DOM/SVG attributes
// client-side, after tokens.css is already loaded.

export const CHART_GRID_STROKE = 'var(--color-border)'

export const CHART_AXIS_TICK = { fill: 'var(--color-text-muted)', fontSize: 12 }

export const CHART_AXIS_LABEL_STYLE = { fill: 'var(--color-text-muted)' }

export const CHART_TOOLTIP_PROPS = {
  contentStyle: {
    background: 'var(--color-bg-raised)',
    border: '1px solid var(--color-border-strong)',
    borderRadius: 'var(--radius-md)',
    color: 'var(--color-text)',
  },
  labelStyle: { color: 'var(--color-text-muted)' },
  itemStyle: { color: 'var(--color-text)' },
}

export const CHART_LEGEND_WRAPPER_STYLE = { color: 'var(--color-text-muted)' }
