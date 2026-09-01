import { useState } from 'react'

function scoreColorClass(score) {
  if (score >= 70) return 'health-badge-green'
  if (score >= 40) return 'health-badge-yellow'
  return 'health-badge-red'
}

const COMPONENT_LABELS = {
  churn_component: 'Churn',
  survival_component: 'Survival',
  segment_component: 'Segment',
  anomaly_component: 'Anomaly',
  clv_component: 'CLV',
}

export default function HealthScoreBadge({ data, loading }) {
  const [expanded, setExpanded] = useState(false)

  if (loading) {
    return <span className="health-badge health-badge-loading">…</span>
  }

  if (!data || data.available === false) {
    return <span className="muted small">n/a</span>
  }

  return (
    <div className="health-badge-wrap">
      <button
        type="button"
        className={`health-badge ${scoreColorClass(data.health_score)}`}
        onClick={() => setExpanded((prev) => !prev)}
        title="Click for component breakdown"
      >
        {Math.round(data.health_score)}
      </button>
      {expanded && (
        <div className="health-breakdown">
          <dl>
            {/* Only renders components this tenant actually has (data.components_used) -
                a tenant missing e.g. a survival model never gets that component at all
                (src/models/health_score.py drops it from the weighted average entirely
                rather than faking a neutral value), so data.components[key] can genuinely
                be undefined for a key in COMPONENT_LABELS. */}
            {(data.components_used || Object.keys(COMPONENT_LABELS)).map((key) => (
              <div key={key} className="health-breakdown-row">
                <dt>{COMPONENT_LABELS[key] || key}</dt>
                <dd>{data.components[key].toFixed(1)}</dd>
              </div>
            ))}
          </dl>
        </div>
      )}
    </div>
  )
}
