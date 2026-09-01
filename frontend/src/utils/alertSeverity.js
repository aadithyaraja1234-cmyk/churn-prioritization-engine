// Single source of truth for alert severity ordering/coloring, shared by
// RecentAlerts (Dashboard) and AlertsPanel (Admin) so both views agree on
// how to sort and color the same /api/alerts data.
const SEVERITY_RANK = { high: 0, medium: 1 }
const SEVERITY_CLASS = { high: 'alert-card-high', medium: 'alert-card-medium' }

export function sortBySeverity(alerts) {
  return [...alerts].sort((a, b) => (SEVERITY_RANK[a.severity] ?? 2) - (SEVERITY_RANK[b.severity] ?? 2))
}

export function severityClass(severity) {
  return SEVERITY_CLASS[severity] || 'alert-card-medium'
}
