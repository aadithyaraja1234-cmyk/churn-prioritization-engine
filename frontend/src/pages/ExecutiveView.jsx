import AppHeader from '../components/AppHeader'
import InterventionRateCaveat from '../components/InterventionRateCaveat'
import Section from '../components/Section'
import { useApiData } from '../hooks/useApiData'

export default function ExecutiveView() {
  const { data: impact, loading: impactLoading, error: impactError } = useApiData('/api/business-impact')
  const { data: queue, loading: queueLoading, error: queueError } = useApiData('/api/action-queue', { limit: 10 })
  const notAvailable = impact?.available === false ? impact.reason : null

  return (
    <div className="dashboard executive-view">
      <AppHeader title="Executive Summary" />

      <Section title="Overview" loading={impactLoading} error={impactError} notAvailable={notAvailable}>
        {impact && !notAvailable && (
          <>
            <div className="kpi-strip">
              <div className="kpi-card kpi-card-primary">
                <span className="kpi-label">Revenue at Risk</span>
                <span className="kpi-value">
                  ${impact.total_revenue_at_risk.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                </span>
              </div>
              <div className="kpi-card">
                <span className="kpi-label">Recoverable Revenue</span>
                <span className="kpi-value">
                  ${impact.total_recoverable_revenue.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                </span>
              </div>
              <div className="kpi-card">
                <span className="kpi-label">Customers Needing Action</span>
                <span className="kpi-value">{impact.customers_needing_action_count}</span>
              </div>
            </div>
            <InterventionRateCaveat metadata={impact.metadata} label="Recoverable Revenue" />
          </>
        )}
      </Section>

      <Section title="Top 10 Recommended Actions" loading={queueLoading} error={queueError} notAvailable={notAvailable}>
        {queue && !notAvailable && queue.action_queue.length === 0 && (
          <p className="muted">No recommended actions right now.</p>
        )}
        {queue && !notAvailable && queue.action_queue.length > 0 && (
          <ol className="opportunity-list">
            {queue.action_queue.slice(0, 10).map((row) => (
              <li key={row.customer_id}>
                <strong>{row.customer_id}</strong> — {row.recommended_action} (potential recovery: $
                {row.expected_revenue_recovery.toFixed(0)})
              </li>
            ))}
          </ol>
        )}
      </Section>
    </div>
  )
}
