import { useEffect, useState } from 'react'
import client from '../api/client'
import Section from './Section'

const WINDOW_LABELS = { '7d': '7 days', '30d': '30 days', '90d': '90 days' }

export default function SurvivalLikelihoodPanel() {
  const [customers, setCustomers] = useState(null)
  const [loadError, setLoadError] = useState(null)
  const [selectedId, setSelectedId] = useState('')

  const [likelihood, setLikelihood] = useState(null)
  const [likelihoodError, setLikelihoodError] = useState(null)
  const [loadingLikelihood, setLoadingLikelihood] = useState(false)

  useEffect(() => {
    let cancelled = false
    client
      .get('/customers')
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

  useEffect(() => {
    if (!selectedId) return
    let cancelled = false
    setLoadingLikelihood(true)
    setLikelihoodError(null)
    client
      .get(`/api/survival/likelihood/${selectedId}`)
      .then((response) => {
        if (!cancelled) setLikelihood(response.data)
      })
      .catch((err) => {
        if (!cancelled) setLikelihoodError(err.response?.data?.detail || err.message)
      })
      .finally(() => {
        if (!cancelled) setLoadingLikelihood(false)
      })
    return () => {
      cancelled = true
    }
  }, [selectedId])

  const notAvailable = likelihood?.available === false ? likelihood.reason : null

  return (
    <Section
      title="Churn Likelihood by Time Window"
      loading={!customers && !loadError}
      error={loadError}
      notAvailable={null}
    >
      {customers && customers.length === 0 && <p className="muted">No customers found.</p>}
      {customers && customers.length > 0 && (
        <label htmlFor="survival-likelihood-select" className="customer-360-select-label">
          Select a customer
          <select
            id="survival-likelihood-select"
            value={selectedId}
            onChange={(event) => setSelectedId(event.target.value)}
          >
            {customers.map((row) => (
              <option key={row.customer_id} value={row.customer_id}>
                {row.customer_id}
              </option>
            ))}
          </select>
        </label>
      )}

      {loadingLikelihood && <p className="muted">Loading…</p>}
      {!loadingLikelihood && likelihoodError && <p className="error-text">Failed to load: {likelihoodError}</p>}
      {!loadingLikelihood && notAvailable && <p className="not-available">{notAvailable}</p>}

      {!loadingLikelihood && likelihood && !notAvailable && (
        <>
          <div className="kpi-strip">
            {Object.entries(WINDOW_LABELS).map(([key, label]) => (
              <div className="kpi-card" key={key}>
                <span className="kpi-label">Likely to churn within {label}</span>
                <span className="kpi-value">{(likelihood[key] * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
          <p className="muted small">
            Estimated from the fitted survival model's own survival function (not just its median), using this
            customer's real contract and tenure — every other feature is held at the training-set median, and
            window days are converted to months using a fixed 30-day month.
          </p>
        </>
      )}
    </Section>
  )
}
