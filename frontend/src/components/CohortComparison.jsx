import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useApiData } from '../hooks/useApiData'
import {
  CHART_AXIS_TICK,
  CHART_GRID_STROKE,
  CHART_LEGEND_WRAPPER_STYLE,
  CHART_TOOLTIP_PROPS,
} from '../styles/chartTheme'
import Section from './Section'

function CohortKpis({ cohort }) {
  if (!cohort) return <p className="not-available">Not available</p>
  return (
    <div className="kpi-strip">
      <div className="kpi-card">
        <span className="kpi-label">ROC-AUC</span>
        <span className="kpi-value">{cohort.roc_auc.toFixed(4)}</span>
      </div>
      <div className="kpi-card">
        <span className="kpi-label">PR-AUC</span>
        <span className="kpi-value">{cohort.pr_auc.toFixed(4)}</span>
      </div>
      <div className="kpi-card">
        <span className="kpi-label">Customers (train/test)</span>
        <span className="kpi-value">
          {cohort.n_train} / {cohort.n_test}
        </span>
      </div>
      <div className="kpi-card">
        <span className="kpi-label">Churn Rate (test)</span>
        <span className="kpi-value">{(cohort.churn_rate_test * 100).toFixed(1)}%</span>
      </div>
    </div>
  )
}

function buildFeatureChartData(telco, banking) {
  const telcoMap = new Map((telco?.top_features || []).map((f) => [f.feature, f.importance]))
  const bankingMap = new Map((banking?.top_features || []).map((f) => [f.feature, f.importance]))
  const allFeatures = [...new Set([...telcoMap.keys(), ...bankingMap.keys()])]
  return allFeatures.map((feature) => ({
    feature,
    Telco: telcoMap.get(feature) || 0,
    Banking: bankingMap.get(feature) || 0,
  }))
}

export default function CohortComparison() {
  const { data, loading, error } = useApiData('/api/cohort-comparison')
  const chartData = data ? buildFeatureChartData(data.telco, data.banking) : null
  const overlapCount = chartData ? chartData.filter((row) => row.Telco > 0 && row.Banking > 0).length : 0

  return (
    <Section title="Cohort Comparison: Telco vs Banking" loading={loading} error={error} notAvailable={null}>
      {data && (
        <div className="cohort-comparison">
          <div className="cohort-columns">
            <div className="cohort-column">
              <h3>Telco</h3>
              <CohortKpis cohort={data.telco} />
            </div>
            <div className="cohort-column">
              <h3>Banking</h3>
              <CohortKpis cohort={data.banking} />
            </div>
          </div>

          {chartData && chartData.length > 0 && (
            <>
              <p className="caveat">
                {overlapCount === 0
                  ? "Zero overlap between the two tenants' top 5 features — Telco's churn drivers (contract, pricing) "
                    + 'and Banking\'s (age, product count, activity) are structurally different populations, not a data issue.'
                  : `${overlapCount} feature(s) appear in both tenants' top 5.`}
              </p>
              <ResponsiveContainer width="100%" height={Math.max(300, chartData.length * 32)}>
                <BarChart data={chartData} layout="vertical" margin={{ left: 110 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID_STROKE} />
                  <XAxis type="number" tick={CHART_AXIS_TICK} />
                  <YAxis type="category" dataKey="feature" width={150} tick={CHART_AXIS_TICK} />
                  <Tooltip {...CHART_TOOLTIP_PROPS} />
                  <Legend wrapperStyle={CHART_LEGEND_WRAPPER_STYLE} />
                  {/* Telco/Banking were hardcoded blue/orange (#2563EB/#F97316),
                      unrelated to any token. A 2-series comparison needs a real
                      CVD-distinguishable pair (not adjacent hues) - accent
                      (violet) vs warn (amber) keeps that same blue-vs-warm-hue
                      strategy BacktestChart already validates, reusing existing
                      tokens instead of two more one-off hex values. */}
                  <Bar dataKey="Telco" fill="var(--color-accent)" isAnimationActive={false} />
                  <Bar dataKey="Banking" fill="var(--color-warn)" isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </>
          )}
        </div>
      )}
    </Section>
  )
}
