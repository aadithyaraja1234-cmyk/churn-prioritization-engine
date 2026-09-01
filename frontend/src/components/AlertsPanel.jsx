import { useApiData } from '../hooks/useApiData'
import { severityClass, sortBySeverity } from '../utils/alertSeverity'
import Section from './Section'

const MAX_PER_TYPE = 8

function groupByType(alerts) {
  const grouped = {}
  for (const alert of alerts) {
    grouped[alert.type] = grouped[alert.type] || []
    grouped[alert.type].push(alert)
  }
  for (const type of Object.keys(grouped)) {
    grouped[type] = sortBySeverity(grouped[type])
  }
  return grouped
}

export default function AlertsPanel() {
  const { data, loading, error } = useApiData('/api/alerts')
  const notAvailable = data?.available === false ? data.reason : null
  const alerts = Array.isArray(data) ? data : null
  const grouped = alerts ? groupByType(alerts) : null

  return (
    <Section title="Alerts" loading={loading} error={error} notAvailable={notAvailable}>
      {alerts && alerts.length === 0 && <p className="muted">No alerts triggered.</p>}

      {grouped && alerts.length > 0 && (
        <div className="alerts-panel">
          <div className="alerts-summary">
            {Object.entries(grouped).map(([type, items]) => (
              <span key={type} className="alerts-summary-chip">
                {items.length} {type.replaceAll('_', ' ')}
              </span>
            ))}
          </div>
          <div className="alerts-list">
            {Object.entries(grouped).map(([type, items]) => {
              const shown = items.slice(0, MAX_PER_TYPE)
              const remaining = items.length - shown.length
              return (
                <div key={type} className="alerts-group">
                  {shown.map((alert, idx) => (
                    <div key={idx} className={`alert-card ${severityClass(alert.severity)}`}>
                      <span className="alert-severity">{alert.severity.toUpperCase()}</span>
                      <p>{alert.description}</p>
                    </div>
                  ))}
                  {remaining > 0 && (
                    <p className="muted small">
                      +{remaining} more {type.replaceAll('_', ' ')} alerts not shown
                    </p>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}
    </Section>
  )
}
