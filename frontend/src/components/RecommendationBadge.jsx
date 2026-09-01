import { useState } from 'react'

export default function RecommendationBadge({ data, loading }) {
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
        className={`recommend-badge ${data.is_priority ? 'recommend-badge-priority' : ''}`}
        onClick={() => setExpanded((prev) => !prev)}
        title="Click for recommended action"
      >
        {data.is_priority ? '★ Priority' : 'View'}
      </button>
      {expanded && (
        <div className="health-breakdown recommend-breakdown">
          <p className="recommend-action-text">{data.recommended_action}</p>
          {data.triggered_by.length > 0 && (
            <p className="muted small">Triggered by: {data.triggered_by.join(', ')}</p>
          )}
        </div>
      )}
    </div>
  )
}
