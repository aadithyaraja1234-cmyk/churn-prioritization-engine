import { useState } from 'react'
import client from '../api/client'
import AppHeader from '../components/AppHeader'
import BudgetCostCaveat from '../components/BudgetCostCaveat'
import BudgetOptimizerForm from '../components/BudgetOptimizerForm'
import CoverageBar from '../components/CoverageBar'
import Section from '../components/Section'
import { useToast } from '../hooks/useToast'

export default function BudgetOptimizer() {
  const { showToast } = useToast()
  const [running, setRunning] = useState(false)
  const [error, setError] = useState(null)
  const [notAvailable, setNotAvailable] = useState(null)
  const [result, setResult] = useState(null)

  async function handleRun({ budgetAmount, costPerIntervention, stopAtPositiveRoi }) {
    setRunning(true)
    setError(null)
    setResult(null)
    setNotAvailable(null)
    try {
      const response = await client.post('/api/budget-optimizer', {
        budget_amount: budgetAmount,
        cost_per_intervention: costPerIntervention,
        stop_at_positive_roi: stopAtPositiveRoi,
      })
      if (response.data.available === false) {
        setNotAvailable(response.data.reason)
      } else {
        setResult(response.data)
        showToast('Budget optimization completed')
      }
    } catch (err) {
      const detail = err.response?.data?.detail
      setError(typeof detail === 'string' ? detail : detail?.error || err.message)
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="dashboard">
      <AppHeader title="Retention Budget Optimizer" />

      <Section title="Optimize a Budget" loading={false} error={null} notAvailable={notAvailable}>
        <BudgetOptimizerForm onRun={handleRun} running={running} />
        {error && <p className="error-text">{error}</p>}
      </Section>

      {running && (
        <section className="dashboard-section">
          <p className="muted">Ranking customers by opportunity score and allocating the budget…</p>
        </section>
      )}

      {result && !running && (
        <section className="dashboard-section">
          <h2>Results</h2>

          <div className="kpi-strip">
            <div className="kpi-card">
              <span className="kpi-label">Customers Covered</span>
              <span className="kpi-value">{result.n_customers_covered}</span>
            </div>
            <div className="kpi-card">
              <span className="kpi-label">Total Recoverable Revenue</span>
              <span className="kpi-value">
                ${result.total_recoverable_revenue.toLocaleString(undefined, { maximumFractionDigits: 0 })}
              </span>
            </div>
            <div className="kpi-card">
              <span className="kpi-label">ROI</span>
              <span className="kpi-value">{(result.roi * 100).toFixed(0)}%</span>
            </div>
            <div className="kpi-card">
              <span className="kpi-label">% of Revenue at Risk Covered</span>
              <span className="kpi-value">{(result.pct_of_at_risk_revenue_covered * 100).toFixed(1)}%</span>
            </div>
            <div className="kpi-card">
              <span className="kpi-label">Total Cost</span>
              <span className="kpi-value">${result.total_cost.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
            </div>
          </div>

          <CoverageBar pct={result.pct_of_at_risk_revenue_covered} />

          {!result.budget_fully_utilized && result.n_customers_covered === 0 && (
            <p className="caveat unspent-budget-note">
              We recommend not spending any of this ${result.budget_amount.toLocaleString(undefined, { maximumFractionDigits: 0 })}{' '}
              budget on interventions — at ${result.cost_per_intervention.toFixed(0)} per intervention, not even the
              single highest-opportunity customer's recoverable revenue clears the cost given current assumptions.
              Try a lower cost-per-intervention estimate, or switch to "Spend entire budget" if you want to fund
              interventions anyway.
            </p>
          )}

          {!result.budget_fully_utilized && result.n_customers_covered > 0 && (
            <p className="caveat unspent-budget-note">
              We recommend spending only $
              {result.total_cost.toLocaleString(undefined, { maximumFractionDigits: 0 })} of your $
              {result.budget_amount.toLocaleString(undefined, { maximumFractionDigits: 0 })} budget — the
              remaining $
              {result.unspent_budget.toLocaleString(undefined, { maximumFractionDigits: 0 })} isn't cost-effective
              to spend given current assumptions (the next customer's recoverable revenue wouldn't cover the $
              {result.cost_per_intervention.toFixed(0)} intervention cost).
            </p>
          )}

          <BudgetCostCaveat result={result} />

          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Customer ID</th>
                  <th>Opportunity Score</th>
                  <th>Recoverable Revenue</th>
                  <th>Churn Probability</th>
                  <th>Recommended Action</th>
                </tr>
              </thead>
              <tbody>
                {result.selected_customers.map((row) => (
                  <tr key={row.customer_id}>
                    <td>{row.customer_id}</td>
                    <td>{row.opportunity_score.toFixed(2)}</td>
                    <td>${row.recoverable_revenue.toFixed(2)}</td>
                    <td>{(row.churn_probability * 100).toFixed(0)}%</td>
                    <td>{row.recommended_action}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {result.selected_customers.length === 0 && (
              <p className="muted">Budget too small to fund even one intervention at this cost per intervention.</p>
            )}
          </div>
        </section>
      )}
    </div>
  )
}
