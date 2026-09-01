import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useApiData } from '../hooks/useApiData'
import { CHART_AXIS_TICK, CHART_GRID_STROKE, CHART_TOOLTIP_PROPS } from '../styles/chartTheme'
import Section from './Section'

// Two mutually exclusive backends for the same section, never both real at
// once for a tenant (see src/models/tenant_training.py's _run_optional_
// modules() - clv and clv_estimated are set up so exactly one of them can
// ever pass for a given tenant): /api/clv/importance is a real, trained
// regression against a genuine per-customer CLV column; /api/clv/estimate
// is a formula-based PROXY for a tenant with no such column at all. Both
// are fetched; whichever one actually has real data wins the render -
// falling back to the "not available" message only when neither does.
export default function ClvSection() {
  const trained = useApiData('/api/clv/importance')
  const estimated = useApiData('/api/clv/estimate')

  const loading = trained.loading || estimated.loading
  const error = trained.error || estimated.error

  const trainedAvailable = trained.data && trained.data.available !== false
  const estimatedAvailable = !trainedAvailable && estimated.data && estimated.data.available !== false
  const notAvailable =
    !trainedAvailable && !estimatedAvailable
      ? trained.data?.reason || estimated.data?.reason || null
      : null

  const chartData =
    trainedAvailable && !error
      ? [...trained.data.feature_importances].sort((a, b) => a.importance - b.importance)
      : null

  return (
    <Section title="Customer Lifetime Value (CLV)" loading={loading} error={error} notAvailable={notAvailable}>
      {trainedAvailable && (
        <>
          <div className="kpi-strip">
            <div className="kpi-card">
              <span className="kpi-label">R²</span>
              <span className="kpi-value">{trained.data.r2.toFixed(3)}</span>
            </div>
            <div className="kpi-card">
              <span className="kpi-label">RMSE</span>
              <span className="kpi-value">{trained.data.rmse.toFixed(1)}</span>
            </div>
            <div className="kpi-card">
              <span className="kpi-label">MAE</span>
              <span className="kpi-value">{trained.data.mae.toFixed(1)}</span>
            </div>
          </div>
          <p className="caveat">
            R² of {trained.data.r2.toFixed(3)} means the model explains only about{' '}
            {Math.round(trained.data.r2 * 100)}% of CLTV&apos;s variance — this dataset&apos;s CLTV appears to
            include a substantial proprietary/internal component not captured by standard account features, so
            treat these predictions as directional, not precise.
          </p>
          <ResponsiveContainer width="100%" height={Math.max(280, chartData.length * 28)}>
            <BarChart data={chartData} layout="vertical" margin={{ left: 100 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID_STROKE} />
              <XAxis type="number" tick={CHART_AXIS_TICK} />
              <YAxis type="category" dataKey="feature" width={140} tick={CHART_AXIS_TICK} />
              <Tooltip {...CHART_TOOLTIP_PROPS} />
              {/* was #0891B2 (teal), unrelated to any token - this is a single
                  featured series, so it ties to the accent identity instead. */}
              <Bar dataKey="importance" fill="var(--color-accent)" isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </>
      )}
      {estimatedAvailable && (
        <>
          <p className="not-available" style={{ marginBottom: '0.75rem' }}>
            ESTIMATED, not measured — this tenant has no real, per-customer CLV column, so these figures are a
            formula-based proxy, not a trained model&apos;s prediction.
          </p>
          <div className="kpi-strip">
            <div className="kpi-card">
              <span className="kpi-label">Customers</span>
              <span className="kpi-value">{estimated.data.count.toLocaleString()}</span>
            </div>
            <div className="kpi-card">
              <span className="kpi-label">Mean Estimated CLV</span>
              <span className="kpi-value">${Math.round(estimated.data.mean).toLocaleString()}</span>
            </div>
            <div className="kpi-card">
              <span className="kpi-label">Median Estimated CLV</span>
              <span className="kpi-value">${Math.round(estimated.data.median).toLocaleString()}</span>
            </div>
          </div>
          <p className="caveat">{estimated.data.methodology_note}</p>
          <ResponsiveContainer width="100%" height={Math.max(280, estimated.data.top_customers.length * 28)}>
            <BarChart data={[...estimated.data.top_customers].reverse()} layout="vertical" margin={{ left: 100 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID_STROKE} />
              <XAxis type="number" tick={CHART_AXIS_TICK} />
              <YAxis type="category" dataKey="customer_id" width={140} tick={CHART_AXIS_TICK} />
              <Tooltip {...CHART_TOOLTIP_PROPS} />
              <Bar dataKey="estimated_clv" name="Estimated CLV" fill="var(--color-accent)" isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </>
      )}
    </Section>
  )
}
