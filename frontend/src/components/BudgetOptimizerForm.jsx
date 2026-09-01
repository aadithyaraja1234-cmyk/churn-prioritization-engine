import { useState } from 'react'

const DEFAULT_COST_PER_INTERVENTION = 75

export default function BudgetOptimizerForm({ onRun, running }) {
  const [budgetAmount, setBudgetAmount] = useState(10000)
  const [costOverride, setCostOverride] = useState('')
  const [stopAtPositiveRoi, setStopAtPositiveRoi] = useState(true)

  function handleSubmit(event) {
    event.preventDefault()
    const costPerIntervention = costOverride === '' ? null : Number(costOverride)
    onRun({ budgetAmount: Number(budgetAmount), costPerIntervention, stopAtPositiveRoi })
  }

  return (
    <form className="scenario-builder" onSubmit={handleSubmit}>
      <div className="scenario-builder-row">
        <label htmlFor="budget-amount">
          Budget Amount ($)
          <input
            id="budget-amount"
            type="number"
            min="0"
            step="100"
            value={budgetAmount}
            onChange={(event) => setBudgetAmount(event.target.value)}
          />
        </label>

        <label htmlFor="cost-per-intervention">
          Cost per Intervention ($) — default {DEFAULT_COST_PER_INTERVENTION}
          <input
            id="cost-per-intervention"
            type="number"
            min="0"
            step="1"
            placeholder={`(optional — defaults to $${DEFAULT_COST_PER_INTERVENTION})`}
            value={costOverride}
            onChange={(event) => setCostOverride(event.target.value)}
          />
        </label>
      </div>

      <div className="budget-spend-mode" role="radiogroup" aria-label="Spending mode">
        <label className="budget-spend-mode-option">
          <input
            type="radio"
            name="spend-mode"
            checked={stopAtPositiveRoi}
            onChange={() => setStopAtPositiveRoi(true)}
          />
          Stop when no longer cost-effective (recommended) — skip the rest of the budget once the next
          customer's recoverable revenue wouldn't cover the intervention cost
        </label>
        <label className="budget-spend-mode-option">
          <input
            type="radio"
            name="spend-mode"
            checked={!stopAtPositiveRoi}
            onChange={() => setStopAtPositiveRoi(false)}
          />
          Spend entire budget — keep allocating down the ranked list until the budget runs out, even to
          customers below break-even
        </label>
      </div>

      <button type="submit" disabled={running || !budgetAmount}>
        {running ? 'Optimizing…' : 'Optimize'}
      </button>
    </form>
  )
}
