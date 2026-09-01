import { useApiData } from '../hooks/useApiData'
import Section from './Section'

export default function SegmentCards() {
  const { data, loading, error } = useApiData('/api/segments')
  const notAvailable = data?.available === false ? data.reason : null

  return (
    <Section title="Customer Segments" loading={loading} error={error} notAvailable={notAvailable}>
      {Array.isArray(data) && data.length === 0 && <p className="muted">No segments available.</p>}
      {Array.isArray(data) && data.length > 0 && (
        <div className="card-grid">
          {data.map((cluster) => (
            <div className="info-card" key={cluster.cluster}>
              <h3>{cluster.label}</h3>
              <p className="card-subtitle">
                {cluster.size} customers ({cluster.pct_of_total.toFixed(1)}%)
              </p>
              <dl>
                <dt>Mean tenure</dt>
                <dd>{cluster.mean_tenure.toFixed(1)} months</dd>
                <dt>Mean monthly charges</dt>
                <dd>${cluster.mean_MonthlyCharges.toFixed(2)}</dd>
                <dt>Dominant contract</dt>
                <dd>{cluster.dominant_contract}</dd>
                <dt>Churn rate</dt>
                <dd>{cluster.churn_rate_pct.toFixed(1)}%</dd>
              </dl>
            </div>
          ))}
        </div>
      )}
    </Section>
  )
}
