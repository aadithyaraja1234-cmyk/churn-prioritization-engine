import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import {
  CHART_AXIS_TICK,
  CHART_GRID_STROKE,
  CHART_LEGEND_WRAPPER_STYLE,
  CHART_TOOLTIP_PROPS,
} from '../styles/chartTheme'
import InterventionRateCaveat from './InterventionRateCaveat'

function formatCurrency(n) {
  const value = Number(n)
  const sign = value < 0 ? '-' : ''
  return `${sign}$${Math.abs(value).toLocaleString(undefined, { maximumFractionDigits: 0 })}`
}

export default function ScenarioResults({ result, compact = false }) {
  const segmentChartData = (result.per_segment_breakdown || []).map((row) => ({
    segment: row.label,
    Before: row.before_revenue_at_risk,
    After: row.after_revenue_at_risk,
  }))

  const deltaHighRisk = result.customers_high_risk.delta
  const deltaClass = deltaHighRisk > 0 ? 'whatif-delta-up' : deltaHighRisk < 0 ? 'whatif-delta-down' : 'whatif-delta-flat'

  return (
    <div className="scenario-results">
      <p className="scenario-summary">{result.summary}</p>

      <div className="kpi-strip">
        <div className="kpi-card">
          <span className="kpi-label">Revenue at Risk (Before → After)</span>
          <span className="kpi-value">
            {formatCurrency(result.total_revenue_at_risk.before)} → {formatCurrency(result.total_revenue_at_risk.after)}
          </span>
        </div>
        <div className="kpi-card">
          <span className="kpi-label">High-Risk Customers (Before → After)</span>
          <span className="kpi-value">
            {result.customers_high_risk.before} → {result.customers_high_risk.after}{' '}
            <span className={deltaClass}>
              ({deltaHighRisk > 0 ? '+' : ''}
              {deltaHighRisk})
            </span>
          </span>
        </div>
        <div className="kpi-card">
          <span className="kpi-label">Recoverable Revenue Change</span>
          <span className="kpi-value">
            {result.net_change_in_recoverable_revenue >= 0 ? '+' : ''}
            {formatCurrency(result.net_change_in_recoverable_revenue)}
          </span>
        </div>
      </div>

      {result.metadata && <InterventionRateCaveat metadata={result.metadata} label="Recoverable Revenue Change" />}

      {!compact && segmentChartData.length > 0 && (
        <ResponsiveContainer width="100%" height={Math.max(220, segmentChartData.length * 60)}>
          <BarChart data={segmentChartData} layout="vertical" margin={{ left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID_STROKE} />
            <XAxis type="number" tick={CHART_AXIS_TICK} />
            <YAxis type="category" dataKey="segment" width={260} tick={{ ...CHART_AXIS_TICK, fontSize: 11 }} />
            <Tooltip formatter={(value) => formatCurrency(value)} {...CHART_TOOLTIP_PROPS} />
            <Legend wrapperStyle={CHART_LEGEND_WRAPPER_STYLE} />
            {/* was #94A3B8/#2563EB, unrelated to any token - "Before" reads as
                the muted/baseline series, "After" as the featured result. */}
            <Bar dataKey="Before" fill="var(--color-text-faint)" isAnimationActive={false} />
            <Bar dataKey="After" fill="var(--color-accent)" isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}
