import { useEffect, useState } from 'react'
import client from '../api/client'
import AppHeader from '../components/AppHeader'
import CustomerTimeline from '../components/CustomerTimeline'
import Section from '../components/Section'

export default function Customer360() {
  const [customers, setCustomers] = useState(null)
  const [loadError, setLoadError] = useState(null)
  const [selectedId, setSelectedId] = useState('')

  const [recommendation, setRecommendation] = useState(null)
  const [recommendationError, setRecommendationError] = useState(null)
  const [recommending, setRecommending] = useState(false)
  const [timelineRefreshKey, setTimelineRefreshKey] = useState(0)

  useEffect(() => {
    let cancelled = false
    client
      // recommend_eligible_only=true - recommend_action_for_customer() only
      // ever scores test-split customers (see /customers' own doc note),
      // so an unfiltered list here would offer ~80% of options that error
      // on "Get Recommendation". Narrowing the list removes that trap
      // instead of just labeling it.
      .get('/customers', { params: { recommend_eligible_only: true } })
      .then((response) => {
        if (cancelled) return
        const rows = response.data
        setCustomers(rows)
        if (rows.length > 0) setSelectedId(rows[0].customer_id)
      })
      .catch((err) => {
        if (!cancelled) setLoadError(err.response?.data?.detail || err.message)
      })
    return () => {
      cancelled = true
    }
  }, [])

  function handleSelectCustomer(customerId) {
    setSelectedId(customerId)
    setRecommendation(null)
    setRecommendationError(null)
  }

  async function handleGetRecommendation() {
    setRecommending(true)
    setRecommendationError(null)
    setRecommendation(null)
    try {
      const response = await client.get(`/api/recommend/${selectedId}`, { params: { log_event: true } })
      if (response.data.available === false) {
        setRecommendationError(response.data.reason)
      } else {
        setRecommendation(response.data)
        setTimelineRefreshKey((key) => key + 1)
      }
    } catch (err) {
      setRecommendationError(err.response?.data?.detail || err.message)
    } finally {
      setRecommending(false)
    }
  }

  return (
    <div className="dashboard">
      <AppHeader title="Customer 360" />

      <Section title="Customer" loading={!customers && !loadError} error={loadError} notAvailable={null}>
        {customers && customers.length === 0 && <p className="muted">No customers found.</p>}
        {customers && customers.length > 0 && (
          <label htmlFor="customer-360-select" className="customer-360-select-label">
            Select a customer
            <select
              id="customer-360-select"
              value={selectedId}
              onChange={(event) => handleSelectCustomer(event.target.value)}
            >
              {customers.map((row) => (
                <option key={row.customer_id} value={row.customer_id}>
                  {row.customer_id}
                </option>
              ))}
            </select>
          </label>
        )}
        {customers && customers.length > 0 && (
          <p className="muted small">
            Showing only customers in the model's held-out test set - the ones a recommendation lookup and
            explanation can actually be computed for.
          </p>
        )}
      </Section>

      <Section title="Recommended Action" loading={false} error={null} notAvailable={null}>
        <div className="recommend-lookup">
          <p className="muted small">
            A deliberate lookup - unlike the Priority Table's badges, this explicitly logs a real event to this
            customer's timeline below.
          </p>
          <button type="button" onClick={handleGetRecommendation} disabled={recommending || !selectedId}>
            {recommending ? 'Looking up…' : 'Get Recommendation'}
          </button>
          {recommendationError && <p className="error-text">{recommendationError}</p>}
          {recommendation && (
            <div className="recommend-lookup-result">
              <p className="recommend-action-text">{recommendation.recommended_action}</p>
              {recommendation.triggered_by.length > 0 && (
                <p className="muted small">Triggered by: {recommendation.triggered_by.join(', ')}</p>
              )}
            </div>
          )}
        </div>
      </Section>

      <Section title="Timeline" loading={false} error={null} notAvailable={null}>
        {selectedId && <CustomerTimeline customerId={selectedId} refreshKey={timelineRefreshKey} />}
      </Section>
    </div>
  )
}
