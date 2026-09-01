import { useApiData } from '../hooks/useApiData'
import { severityClass, sortBySeverity } from '../utils/alertSeverity'
import Section from './Section'

const DISPLAY_LIMIT = 5

export default function RecentAlerts() {
  const { data, loading, error } = useApiData('/api/alerts')
  const notAvailable = data?.available === false ? data.reason : null
  const alerts = Array.isArray(data) ? data : null

  const topAlerts = alerts ? sortBySeverity(alerts).slice(0, DISPLAY_LIMIT) : null

  return (
    <Section title="Recent Alerts" loading={loading} error={error} notAvailable={notAvailable}>
      {alerts && alerts.length === 0 && <p className="muted">No alerts triggered.</p>}
      {topAlerts && topAlerts.length > 0 && (
        <div className="alerts-list">
          {topAlerts.map((alert, idx) => (
            <div key={idx} className={`alert-card ${severityClass(alert.severity)}`}>
              <span className="alert-severity">{alert.severity.toUpperCase()}</span>
              <p>{alert.description}</p>
            </div>
          ))}
          {alerts.length > topAlerts.length && (
            <p className="muted small">
              +{alerts.length - topAlerts.length} more alerts — see the full list on the Admin page
            </p>
          )}
        </div>
      )}
    </Section>
  )
}
