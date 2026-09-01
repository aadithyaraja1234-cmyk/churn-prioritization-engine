import { useApiData } from '../hooks/useApiData'
import InterventionRateCaveat from './InterventionRateCaveat'
import Section from './Section'

export default function BusinessImpactKpiStrip() {
  const { data, loading, error } = useApiData('/api/business-impact')
  const notAvailable = data?.available === false ? data.reason : null

  return (
    <Section title="Business Impact" loading={loading} error={error} notAvailable={notAvailable}>
      {data && !notAvailable && (
        <>
          <div className="kpi-strip">
            <div className="kpi-card kpi-card-primary">
              <span className="kpi-label">Revenue at Risk</span>
              <span className="kpi-value">${data.total_revenue_at_risk.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
            </div>
            <div className="kpi-card">
              <span className="kpi-label">Recoverable Revenue</span>
              <span className="kpi-value">${data.total_recoverable_revenue.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
            </div>
            <div className="kpi-card">
              <span className="kpi-label">Customers Needing Action</span>
              <span className="kpi-value">{data.customers_needing_action_count}</span>
            </div>
          </div>
          <InterventionRateCaveat metadata={data.metadata} label="Recoverable Revenue" />
        </>
      )}
    </Section>
  )
}
