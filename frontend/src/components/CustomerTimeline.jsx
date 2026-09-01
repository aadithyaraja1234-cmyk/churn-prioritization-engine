import { useEffect, useState } from 'react'
import client from '../api/client'

const EVENT_ICONS = {
  signup: '◆',
  contract_snapshot: '▪',
  prediction: '▲',
  recommendation: '●',
  scenario: '■',
}

function formatTimestamp(iso) {
  return new Date(iso).toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function EventDetails({ event }) {
  switch (event.event_type) {
    case 'signup':
      return (
        <p>
          Estimated signup date, back-calculated from this customer's current tenure (~{event.details.tenure_months}{' '}
          months).
        </p>
      )
    case 'contract_snapshot':
      return (
        <p>
          Current contract on file: <strong>{event.details.contract}</strong>
          <br />
          <span className="muted small">{event.details.note}</span>
        </p>
      )
    case 'prediction':
      return (
        <p>
          Churn Probability: <strong>{(event.details.churn_probability * 100).toFixed(1)}%</strong>
          <br />
          <span className="muted small">model: {event.details.model_version}</span>
        </p>
      )
    case 'recommendation':
      return (
        <p>
          Recommended action: <strong>{event.details.recommended_action}</strong>
          {event.details.triggered_by.length > 0 && (
            <>
              <br />
              <span className="muted small">triggered by: {event.details.triggered_by.join(', ')}</span>
            </>
          )}
        </p>
      )
    case 'scenario':
      return (
        <p>
          Scenario <strong>{event.details.scenario_name}</strong>: this customer's probability{' '}
          {(event.details.before_probability * 100).toFixed(1)}% → {(event.details.after_probability * 100).toFixed(1)}%
          {!event.details.was_affected && <span className="muted small"> (not directly affected by this scenario)</span>}
        </p>
      )
    default:
      return null
  }
}

export default function CustomerTimeline({ customerId, refreshKey }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!customerId) return
    let cancelled = false
    setData(null)
    setError(null)
    client
      .get(`/api/customer/${customerId}/timeline`)
      .then((response) => {
        if (!cancelled) setData(response.data)
      })
      .catch((err) => {
        if (!cancelled) setError(err.response?.data?.detail || err.message)
      })
    return () => {
      cancelled = true
    }
  }, [customerId, refreshKey])

  if (error) return <p className="error-text">Failed to load timeline: {error}</p>
  if (!data) return <p className="muted">Loading…</p>
  if (data.available === false) return <p className="not-available">{data.reason}</p>

  const realEventCount = data.events.filter((event) => event.data_source === 'real_system_event').length

  return (
    <div className="customer-timeline">
      {realEventCount === 0 && (
        <p className="caveat">
          This customer's activity history will grow as predictions, recommendations, and scenarios are run for
          them — currently showing available snapshot data.
        </p>
      )}

      <ol className="timeline-list">
        {data.events.map((event, index) => (
          <li key={index} className="timeline-event">
            <span className={`timeline-icon timeline-icon-${event.event_type}`}>{EVENT_ICONS[event.event_type] || '•'}</span>
            <div className="timeline-content">
              <div className="timeline-content-header">
                <span className="timeline-type">{event.event_type.replace('_', ' ')}</span>
                <span
                  className={
                    event.data_source === 'real_system_event'
                      ? 'timeline-badge timeline-badge-recorded'
                      : 'timeline-badge timeline-badge-estimated'
                  }
                >
                  {event.data_source === 'real_system_event' ? 'recorded' : 'estimated'}
                </span>
                <span className="muted small timeline-timestamp">{formatTimestamp(event.timestamp)}</span>
              </div>
              <EventDetails event={event} />
            </div>
          </li>
        ))}
      </ol>
    </div>
  )
}
