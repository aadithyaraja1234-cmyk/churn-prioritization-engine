import { useCallback, useEffect, useState } from 'react'
import client from '../api/client'
import AppHeader from '../components/AppHeader'
import Section from '../components/Section'
import ScenarioBuilder from '../components/ScenarioBuilder'
import ScenarioHistory from '../components/ScenarioHistory'
import ScenarioResults from '../components/ScenarioResults'
import { useToast } from '../hooks/useToast'

export default function ScenarioSimulator() {
  const { showToast } = useToast()
  const [notAvailable, setNotAvailable] = useState(null)

  const [running, setRunning] = useState(false)
  const [runError, setRunError] = useState(null)
  const [result, setResult] = useState(null)

  const [history, setHistory] = useState(null)
  const [historyError, setHistoryError] = useState(null)

  const [compareIds, setCompareIds] = useState([])
  const [compareResults, setCompareResults] = useState(null)
  const [compareError, setCompareError] = useState(null)

  const fetchHistory = useCallback(() => {
    client
      .get('/api/scenario/history')
      .then((response) => {
        if (response.data.available === false) {
          setNotAvailable(response.data.reason)
          setHistory([])
        } else {
          setHistory(response.data)
        }
      })
      .catch((err) => setHistoryError(err.response?.data?.detail || err.message))
  }, [])

  useEffect(() => {
    fetchHistory()
  }, [fetchHistory])

  async function handleRun({ scenarioType, params, scenarioName }) {
    setRunning(true)
    setRunError(null)
    setResult(null)
    setCompareResults(null)
    try {
      const response = await client.post('/api/scenario', {
        scenario_type: scenarioType,
        params,
        scenario_name: scenarioName,
      })
      if (response.data.available === false) {
        setNotAvailable(response.data.reason)
      } else {
        setResult(response.data)
        fetchHistory()
        showToast('Scenario run completed')
      }
    } catch (err) {
      const detail = err.response?.data?.detail
      setRunError(typeof detail === 'string' ? detail : detail?.error || err.message)
    } finally {
      setRunning(false)
    }
  }

  async function handleViewHistoryItem(id) {
    setRunError(null)
    setCompareResults(null)
    try {
      const response = await client.get(`/api/scenario/${id}`)
      setResult(response.data)
    } catch (err) {
      setRunError(err.response?.data?.detail || err.message)
    }
  }

  function toggleCompareId(id) {
    setCompareIds((prev) => {
      if (prev.includes(id)) return prev.filter((existing) => existing !== id)
      if (prev.length >= 2) return [prev[1], id]
      return [...prev, id]
    })
  }

  async function handleCompare() {
    setCompareError(null)
    setCompareResults(null)
    try {
      const responses = await Promise.all(compareIds.map((id) => client.get(`/api/scenario/${id}`)))
      setCompareResults(responses.map((response) => response.data))
    } catch (err) {
      setCompareError(err.response?.data?.detail || err.message)
    }
  }

  return (
    <div className="dashboard">
      <AppHeader title="Scenario Simulator" />

      <Section title="Build a Scenario" loading={false} error={null} notAvailable={notAvailable}>
        <ScenarioBuilder onRun={handleRun} running={running} />
        {runError && <p className="error-text">{runError}</p>}
      </Section>

      {running && (
        <section className="dashboard-section">
          <p className="muted">Re-scoring the full customer population under this scenario — this can take a few seconds…</p>
        </section>
      )}

      {result && !running && (
        <section className="dashboard-section">
          <h2>Results</h2>
          <ScenarioResults result={result} />
        </section>
      )}

      <Section
        title="Scenario History"
        loading={!history && !historyError && !notAvailable}
        error={historyError}
        notAvailable={notAvailable}
      >
        {history && (
          <ScenarioHistory
            history={history}
            onView={handleViewHistoryItem}
            compareIds={compareIds}
            onToggleCompare={toggleCompareId}
            onCompare={handleCompare}
          />
        )}
      </Section>

      {compareError && <p className="error-text">{compareError}</p>}
      {compareResults && (
        <section className="dashboard-section">
          <h2>Compare Scenarios</h2>
          <div className="scenario-compare-columns">
            {compareResults.map((row) => (
              <div className="scenario-compare-column" key={row.id}>
                <h3>{row.scenario_name}</h3>
                <ScenarioResults result={row} compact />
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}
