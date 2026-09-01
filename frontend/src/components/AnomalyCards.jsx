import { useApiData } from '../hooks/useApiData'
import Section from './Section'

export default function AnomalyCards() {
  const { data, loading, error } = useApiData('/api/anomalies', { limit: 5 })
  const notAvailable = data?.available === false ? data.reason : null

  return (
    <Section title="Flagged Anomalies" loading={loading} error={error} notAvailable={notAvailable}>
      {Array.isArray(data) && data.length === 0 && <p className="muted">No anomalies flagged.</p>}
      {Array.isArray(data) && data.length > 0 && (
        <div className="card-grid">
          {data.map((customer) => (
            <div className="info-card" key={customer.customerID}>
              <h3>{customer.customerID}</h3>
              <dl>
                <dt>Tenure</dt>
                <dd>{customer.tenure} months</dd>
                <dt>Monthly Charges</dt>
                <dd>${customer.MonthlyCharges?.toFixed(2)}</dd>
                <dt>Contract</dt>
                <dd>{customer.Contract}</dd>
                <dt>Internet Service</dt>
                <dd>{customer.InternetService}</dd>
                <dt>Phone Service</dt>
                <dd>{customer.PhoneService}</dd>
                <dt>Anomaly Score</dt>
                <dd>{customer.anomaly_score.toFixed(4)}</dd>
              </dl>
            </div>
          ))}
        </div>
      )}
    </Section>
  )
}
