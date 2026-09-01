const SCENARIO_TYPE_LABELS = {
  uniform_charge_change: 'Price Change',
  contract_migration: 'Contract Migration',
}

export default function ScenarioHistory({ history, onView, compareIds, onToggleCompare, onCompare }) {
  if (history.length === 0) {
    return <p className="muted">No scenarios run yet — build one above to get started.</p>
  }

  return (
    <div className="scenario-history">
      <ul className="scenario-history-list">
        {history.map((row) => (
          <li key={row.id} className="scenario-history-item">
            <input
              type="checkbox"
              className="scenario-history-checkbox"
              checked={compareIds.includes(row.id)}
              onChange={() => onToggleCompare(row.id)}
              aria-label={`Select ${row.scenario_name} for comparison`}
            />
            <button type="button" className="scenario-history-link" onClick={() => onView(row.id)}>
              <strong>{row.scenario_name}</strong>
              <span className="muted small"> ({SCENARIO_TYPE_LABELS[row.scenario_type] || row.scenario_type})</span>
              <div className="muted small">
                Revenue at Risk change: {row.headline_delta >= 0 ? '+' : '-'}$
                {Math.abs(row.headline_delta).toLocaleString(undefined, { maximumFractionDigits: 0 })}
              </div>
            </button>
          </li>
        ))}
      </ul>
      <button
        type="button"
        className="scenario-history-compare-button"
        disabled={compareIds.length !== 2}
        onClick={onCompare}
      >
        Compare Selected ({compareIds.length}/2)
      </button>
    </div>
  )
}
