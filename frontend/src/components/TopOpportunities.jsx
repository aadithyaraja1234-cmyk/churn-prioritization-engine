import { useApiData } from '../hooks/useApiData'
import Section from './Section'

const DISPLAY_LIMIT = 10

export default function TopOpportunities() {
  const { data, loading, error } = useApiData('/api/business-impact')
  const notAvailable = data?.available === false ? data.reason : null

  const rows = data && !notAvailable ? data.customers.slice(0, DISPLAY_LIMIT) : null

  return (
    <Section title="Top 10 Opportunities" loading={loading} error={error} notAvailable={notAvailable}>
      {rows && rows.length === 0 && <p className="muted">No opportunities to show right now.</p>}
      {rows && rows.length > 0 && (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Customer ID</th>
                <th>Revenue at Risk</th>
                <th>Recoverable Revenue</th>
                <th>Opportunity Score</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.customerID}>
                  <td>{row.customerID}</td>
                  <td>${row.revenue_at_risk.toFixed(2)}</td>
                  <td>${row.recoverable_revenue.toFixed(2)}</td>
                  <td>{row.opportunity_score.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Section>
  )
}
