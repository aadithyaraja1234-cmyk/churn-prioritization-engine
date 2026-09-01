import { useEffect, useState } from 'react'
import client from '../api/client'
import Section from './Section'

let nextRowId = 0
function newRow(feature = '', value = '') {
  return { id: nextRowId++, feature, value }
}

// No hardcoded Telco columns here on purpose - this used to fix its override
// fields to Contract/tenure/MonthlyCharges/InternetService with hardcoded
// Contract/InternetService dropdown options, which only ever worked for
// Telco (and any other tenant that happened to reuse Telco's exact schema).
// Real feature names come from /api/model/importance (already tenant-generic -
// same real feature_names split_indices.json saved at training time for
// whichever columns THIS tenant actually mapped), and each row's starting
// value is pre-filled from the selected customer's own real raw_features, so
// the user edits a real value they can see, not a guess.
export default function WhatIfPanel() {
  const [customers, setCustomers] = useState(null)
  const [loadError, setLoadError] = useState(null)
  const [featureNames, setFeatureNames] = useState(null)

  const [selectedId, setSelectedId] = useState('')
  const [selectedCustomer, setSelectedCustomer] = useState(null)
  const [rows, setRows] = useState([newRow()])

  const [result, setResult] = useState(null)
  const [simError, setSimError] = useState(null)
  const [validFeatures, setValidFeatures] = useState(null)
  const [validValues, setValidValues] = useState(null)
  const [simulating, setSimulating] = useState(false)

  useEffect(() => {
    let cancelled = false
    Promise.all([
      client.get('/customers', { params: { recommend_eligible_only: true } }),
      client.get('/api/model/importance'),
    ])
      .then(([customersResponse, importanceResponse]) => {
        if (cancelled) return
        const rowsData = customersResponse.data
        setCustomers(rowsData)
        if (rowsData.length > 0) selectCustomer(rowsData[0])
        if (Array.isArray(importanceResponse.data)) {
          setFeatureNames(importanceResponse.data.map((entry) => entry.feature))
        }
      })
      .catch((err) => {
        if (!cancelled) setLoadError(err.response?.data?.detail || err.message)
      })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function selectCustomer(row) {
    setSelectedId(row.customer_id)
    setSelectedCustomer(row)
    setRows([newRow()])
    setResult(null)
    setSimError(null)
    setValidFeatures(null)
    setValidValues(null)
  }

  function handleCustomerChange(event) {
    const row = customers.find((c) => c.customer_id === event.target.value)
    if (row) selectCustomer(row)
  }

  function updateRow(id, patch) {
    setRows((prev) => prev.map((row) => (row.id === id ? { ...row, ...patch } : row)))
  }

  function handleFeatureChange(id, feature) {
    // Pre-fill with the customer's own current value for that feature, so
    // the user sees what they're actually changing from.
    const currentValue = selectedCustomer?.raw_features?.[feature]
    updateRow(id, { feature, value: currentValue === undefined || currentValue === null ? '' : String(currentValue) })
  }

  function addRow() {
    setRows((prev) => [...prev, newRow()])
  }

  function removeRow(id) {
    setRows((prev) => (prev.length > 1 ? prev.filter((row) => row.id !== id) : prev))
  }

  async function handleSimulate() {
    setSimulating(true)
    setSimError(null)
    setResult(null)
    setValidFeatures(null)
    setValidValues(null)

    const overrides = {}
    for (const row of rows) {
      if (!row.feature) continue
      const originalValue = selectedCustomer?.raw_features?.[row.feature]
      // Send back a number when this feature's real value is numeric (so a
      // numeric override doesn't get rejected as a string), otherwise as-is.
      overrides[row.feature] = typeof originalValue === 'number' ? Number(row.value) : row.value
    }

    try {
      const response = await client.post('/api/whatif', { customer_id: selectedId, overrides })
      if (response.data.available === false) {
        setSimError(response.data.reason)
      } else {
        setResult(response.data)
      }
    } catch (err) {
      const detail = err.response?.data?.detail
      if (typeof detail === 'object' && detail !== null) {
        setSimError(detail.error)
        if (detail.valid_features) setValidFeatures(detail.valid_features)
        if (detail.valid_values) setValidValues(detail.valid_values)
      } else {
        setSimError(detail || err.message)
      }
    } finally {
      setSimulating(false)
    }
  }

  const delta = result ? result.delta : null
  const deltaDirection = delta == null ? null : delta < -0.001 ? 'down' : delta > 0.001 ? 'up' : 'flat'
  const hasOverride = rows.some((row) => row.feature)

  return (
    <Section title="What-If Simulator" loading={!customers && !loadError} error={loadError} notAvailable={null}>
      {customers && (
        <div className="whatif-panel">
          <div className="whatif-controls">
            <label htmlFor="whatif-customer">
              Customer
              <select id="whatif-customer" value={selectedId} onChange={handleCustomerChange}>
                {customers.map((row) => (
                  <option key={row.customer_id} value={row.customer_id}>
                    {row.customer_id}
                  </option>
                ))}
              </select>
            </label>

            {rows.map((row) => (
              <div className="whatif-override-row" key={row.id}>
                <label htmlFor={`whatif-feature-${row.id}`}>
                  Feature
                  <select
                    id={`whatif-feature-${row.id}`}
                    value={row.feature}
                    onChange={(event) => handleFeatureChange(row.id, event.target.value)}
                  >
                    <option value="">— choose a feature —</option>
                    {(featureNames || []).map((name) => (
                      <option key={name} value={name}>
                        {name}
                      </option>
                    ))}
                  </select>
                </label>
                <label htmlFor={`whatif-value-${row.id}`}>
                  New value
                  <input
                    id={`whatif-value-${row.id}`}
                    type="text"
                    value={row.value}
                    disabled={!row.feature}
                    onChange={(event) => updateRow(row.id, { value: event.target.value })}
                  />
                </label>
                {rows.length > 1 && (
                  <button type="button" className="whatif-remove-row" onClick={() => removeRow(row.id)}>
                    Remove
                  </button>
                )}
              </div>
            ))}
            <button type="button" onClick={addRow} className="whatif-add-row">
              + Add another override
            </button>

            <button type="button" onClick={handleSimulate} disabled={simulating || !selectedId || !hasOverride}>
              {simulating ? 'Simulating…' : 'Simulate'}
            </button>
          </div>

          {simError && (
            <div className="error-text">
              <p>{simError}</p>
              {validFeatures && <p className="muted small">Valid features: {validFeatures.join(', ')}</p>}
              {validValues && <p className="muted small">Valid values: {validValues.join(', ')}</p>}
            </div>
          )}

          {result && (
            <div className="whatif-result">
              <div className="whatif-prob">
                <span className="kpi-label">Original</span>
                <span className="kpi-value">{(result.original_probability * 100).toFixed(1)}%</span>
              </div>
              <div className={`whatif-delta whatif-delta-${deltaDirection}`}>
                {deltaDirection === 'down' ? '▼' : deltaDirection === 'up' ? '▲' : '—'}{' '}
                {Math.abs(result.delta * 100).toFixed(1)} pts
              </div>
              <div className="whatif-prob">
                <span className="kpi-label">New</span>
                <span className="kpi-value">{(result.new_probability * 100).toFixed(1)}%</span>
              </div>
            </div>
          )}
        </div>
      )}
    </Section>
  )
}
