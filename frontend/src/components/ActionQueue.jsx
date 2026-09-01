import { useApiData } from '../hooks/useApiData'
import { useToast } from '../hooks/useToast'
import { downloadCsv } from '../utils/csvExport'
import InterventionRateCaveat from './InterventionRateCaveat'
import Section from './Section'

export default function ActionQueue() {
  const { data, loading, error } = useApiData('/api/action-queue', { limit: 20 })
  const notAvailable = data?.available === false ? data.reason : null
  const queue = data && !notAvailable ? data.action_queue : null
  const { showToast } = useToast()

  function handleExport() {
    const headers = ['Customer ID', 'Recommended Action', 'Expected Revenue Recovery', 'Opportunity Score']
    const csvRows = queue.map((row) => [
      row.customer_id,
      row.recommended_action,
      row.expected_revenue_recovery.toFixed(2),
      row.opportunity_score.toFixed(2),
    ])
    downloadCsv('action_queue.csv', headers, csvRows)
    showToast(`Exported ${csvRows.length} rows to CSV`)
  }

  return (
    <Section title="Action Queue" loading={loading} error={error} notAvailable={notAvailable}>
      {queue && queue.length === 0 && <p className="muted">No items in the action queue right now.</p>}
      {queue && queue.length > 0 && (
        <>
          <InterventionRateCaveat metadata={data.metadata} label="Expected Revenue Recovery" />
          <div className="table-wrap">
            <button type="button" className="export-csv-button" onClick={handleExport}>
              Export to CSV
            </button>
            <table>
              <thead>
                <tr>
                  <th>Customer ID</th>
                  <th>Recommended Action</th>
                  <th>Expected Revenue Recovery</th>
                  <th>Opportunity Score</th>
                </tr>
              </thead>
              <tbody>
                {queue.map((row) => (
                  <tr key={row.customer_id}>
                    <td>{row.customer_id}</td>
                    <td>{row.recommended_action}</td>
                    <td>${row.expected_revenue_recovery.toFixed(2)}</td>
                    <td>{row.opportunity_score.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Section>
  )
}
