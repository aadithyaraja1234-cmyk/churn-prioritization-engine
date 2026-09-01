import { useEffect, useMemo, useState } from 'react'
import client from '../api/client'
import { useApiData } from '../hooks/useApiData'
import { useToast } from '../hooks/useToast'
import { downloadCsv } from '../utils/csvExport'
import HealthScoreBadge from './HealthScoreBadge'
import RecommendationBadge from './RecommendationBadge'
import Section from './Section'
import TrackingFields from './TrackingFields'

// Kept at 20, not 50: each row fires a health-score AND a recommend call,
// and recommend's per-customer cost (explain_customer()'s marginal-effect
// loop, ~0.15s of real CPU-bound sklearn inference) isn't cached across
// requests. 50 rows measured at up to ~25s for the slowest of the batch to
// resolve on a single-worker uvicorn process; 20 keeps this responsive.
const DISPLAY_LIMIT = 20

const COLUMNS = [
  { key: 'customerID', label: 'Customer ID' },
  { key: 'revenue_at_risk', label: 'Revenue at Risk' },
  { key: 'recoverable_revenue', label: 'Recoverable Revenue' },
  {
    key: 'opportunity_score',
    label: 'Opportunity Score',
    tooltip:
      'Opportunity Score — a prioritization heuristic (revenue × ease-of-saving × value), ' +
      'distinct from the backtested revenue-weighted strategy shown in the Backtest chart above.',
  },
  { key: 'confidence', label: 'Confidence' },
]

export default function PriorityTable() {
  const { data, loading, error } = useApiData('/api/business-impact')
  const notAvailable = data?.available === false ? data.reason : null
  const [sortKey, setSortKey] = useState('opportunity_score')
  const [sortAsc, setSortAsc] = useState(false)
  const [healthScores, setHealthScores] = useState({})
  const [recommendations, setRecommendations] = useState({})
  const { showToast } = useToast()

  // /api/business-impact already returns all test-set customers sorted by
  // opportunity_score - only fetch health-score/recommend for the slice we
  // actually display, so this stays bounded regardless of test-set size.
  const topRows = useMemo(() => {
    if (!data || !Array.isArray(data.customers)) return null
    return data.customers.slice(0, DISPLAY_LIMIT)
  }, [data])

  const rows = useMemo(() => {
    if (!topRows) return null
    return [...topRows].sort((a, b) => {
      const left = a[sortKey]
      const right = b[sortKey]
      if (left < right) return sortAsc ? -1 : 1
      if (left > right) return sortAsc ? 1 : -1
      return 0
    })
  }, [topRows, sortKey, sortAsc])

  useEffect(() => {
    if (!topRows) return
    let cancelled = false
    Promise.all(
      topRows.map((row) =>
        client
          .get(`/api/health-score/${row.customerID}`)
          .then((response) => [row.customerID, response.data])
          .catch(() => [row.customerID, null]),
      ),
    ).then((entries) => {
      if (!cancelled) setHealthScores(Object.fromEntries(entries))
    })
    return () => {
      cancelled = true
    }
  }, [topRows])

  useEffect(() => {
    if (!topRows) return
    let cancelled = false
    Promise.all(
      topRows.map((row) =>
        client
          .get(`/api/recommend/${row.customerID}`)
          .then((response) => [row.customerID, response.data])
          .catch(() => [row.customerID, null]),
      ),
    ).then((entries) => {
      if (!cancelled) setRecommendations(Object.fromEntries(entries))
    })
    return () => {
      cancelled = true
    }
  }, [topRows])

  function handleSort(key) {
    if (key === sortKey) {
      setSortAsc((prev) => !prev)
    } else {
      setSortKey(key)
      setSortAsc(false)
    }
  }

  function handleExport() {
    const headers = [
      'Customer ID',
      'Revenue at Risk',
      'Recoverable Revenue',
      'Opportunity Score',
      'Confidence',
      'Recommended Action',
      'Health Score',
    ]
    const csvRows = rows.map((row) => [
      row.customerID,
      row.revenue_at_risk.toFixed(2),
      row.recoverable_revenue.toFixed(2),
      row.opportunity_score.toFixed(2),
      (row.confidence * 100).toFixed(0) + '%',
      recommendations[row.customerID]?.recommended_action ?? '',
      healthScores[row.customerID]?.health_score ?? '',
    ])
    downloadCsv('priority_table.csv', headers, csvRows)
    showToast(`Exported ${csvRows.length} rows to CSV`)
  }

  return (
    <Section title="Priority Table (Top 20 Opportunities)" loading={loading} error={error} notAvailable={notAvailable}>
      {rows && rows.length === 0 && <p className="muted">No customers to prioritize right now.</p>}
      {rows && rows.length > 0 && (
        <div className="table-wrap">
          <button type="button" className="export-csv-button" onClick={handleExport}>
            Export to CSV
          </button>
          <table>
            <thead>
              <tr>
                {COLUMNS.map((col) => (
                  <th key={col.key} onClick={() => handleSort(col.key)}>
                    {col.tooltip ? (
                      <span className="th-tooltip-trigger">
                        {col.label}
                        {sortKey === col.key ? (sortAsc ? ' ▲' : ' ▼') : ''}
                        <span className="th-tooltip-bubble">{col.tooltip}</span>
                      </span>
                    ) : (
                      <>
                        {col.label}
                        {sortKey === col.key ? (sortAsc ? ' ▲' : ' ▼') : ''}
                      </>
                    )}
                  </th>
                ))}
                <th>Suggested Action</th>
                <th>Health Score</th>
                <th>
                  <span className="th-tooltip-trigger">
                    Tracking (manual — you fill these in)
                    <span className="th-tooltip-bubble">
                      Assigned manager, status, and call date are entered by you, not generated or suggested by the
                      system.
                    </span>
                  </span>
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.customerID}>
                  <td>{row.customerID}</td>
                  <td>${row.revenue_at_risk.toFixed(2)}</td>
                  <td>${row.recoverable_revenue.toFixed(2)}</td>
                  <td>{row.opportunity_score.toFixed(2)}</td>
                  <td>{(row.confidence * 100).toFixed(0)}%</td>
                  <td>
                    <RecommendationBadge
                      data={recommendations[row.customerID]}
                      loading={!(row.customerID in recommendations)}
                    />
                  </td>
                  <td>
                    <HealthScoreBadge
                      data={healthScores[row.customerID]}
                      loading={!(row.customerID in healthScores)}
                    />
                  </td>
                  <td>
                    <TrackingFields customerId={row.customerID} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Section>
  )
}
