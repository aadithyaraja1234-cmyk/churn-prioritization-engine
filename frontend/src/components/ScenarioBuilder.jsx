import { useEffect, useState } from 'react'
import client from '../api/client'

export default function ScenarioBuilder({ onRun, running }) {
  const [scenarioType, setScenarioType] = useState('uniform_charge_change')
  const [scenarioName, setScenarioName] = useState('')
  const [percentChange, setPercentChange] = useState(10)
  const [fromContract, setFromContract] = useState('')
  const [toContract, setToContract] = useState('')
  const [migrationRate, setMigrationRate] = useState(0.3)
  const [discountPercent, setDiscountPercent] = useState(10)
  const [targetSegment, setTargetSegment] = useState('')
  const [adoptionRate, setAdoptionRate] = useState(0.3)
  const [segments, setSegments] = useState([])
  // The tenant's real segment/contract-equivalent category values (e.g.
  // Telco: "Month-to-month"/"One year"/"Two year"; Aurora:
  // "Basic"/"Plus"/"Premium") - NOT a fixed Telco literal list, since
  // contract_migration's from/to values must be real categories for
  // whichever column this tenant's survival module actually resolved as
  // its segment_feature_column (see scenario.py's NoSegmentFeatureColumnError -
  // structurally, this option only ever works when survival has already
  // passed its own gate for this tenant, which is exactly when
  // /api/survival/segments has real keys to offer here).
  const [contractOptions, setContractOptions] = useState(null)

  useEffect(() => {
    client
      .get('/api/segments')
      .then((response) => {
        if (Array.isArray(response.data)) setSegments(response.data)
      })
      .catch(() => {
        // Segment list is only needed for the discount_offer target_segment
        // dropdown - if it fails to load, "All customers" still works.
      })

    client
      .get('/api/survival/segments')
      .then((response) => {
        const options = response.data?.available === false ? [] : Object.keys(response.data || {})
        setContractOptions(options)
        if (options.length >= 2) {
          setFromContract(options[0])
          setToContract(options[options.length - 1])
        }
      })
      .catch(() => setContractOptions([]))
  }, [])

  function handleSubmit(event) {
    event.preventDefault()

    let autoName
    let params
    if (scenarioType === 'uniform_charge_change') {
      autoName = `${percentChange > 0 ? '+' : ''}${percentChange}% price change`
      params = { percent_change: Number(percentChange) }
    } else if (scenarioType === 'contract_migration') {
      autoName = `${fromContract} → ${toContract} (${Math.round(migrationRate * 100)}%)`
      params = { from_contract: fromContract, to_contract: toContract, migration_rate: Number(migrationRate) }
    } else if (scenarioType === 'discount_offer') {
      const segmentDesc = targetSegment === '' ? 'all customers' : `segment ${targetSegment}`
      autoName = `${discountPercent}% discount to ${segmentDesc}`
      params = {
        discount_percent: Number(discountPercent),
        target_segment: targetSegment === '' ? null : Number(targetSegment),
      }
    } else {
      autoName = `Loyalty program (${Math.round(adoptionRate * 100)}% adoption)`
      params = { adoption_rate: Number(adoptionRate) }
    }

    onRun({ scenarioType, params, scenarioName: scenarioName.trim() || autoName })
  }

  return (
    <form className="scenario-builder" onSubmit={handleSubmit}>
      <div className="scenario-builder-row">
        <label htmlFor="scenario-type">
          Scenario Type
          <select id="scenario-type" value={scenarioType} onChange={(event) => setScenarioType(event.target.value)}>
            <option value="uniform_charge_change">Price Change</option>
            <option value="contract_migration" disabled={contractOptions !== null && contractOptions.length < 2}>
              Contract Migration{contractOptions !== null && contractOptions.length < 2 ? ' (unavailable for this tenant)' : ''}
            </option>
            <option value="discount_offer">Discount Offer</option>
            <option value="loyalty_program">Loyalty Program</option>
          </select>
        </label>

        <label htmlFor="scenario-name">
          Scenario Name
          <input
            id="scenario-name"
            type="text"
            placeholder="(optional — auto-generated if left blank)"
            value={scenarioName}
            onChange={(event) => setScenarioName(event.target.value)}
          />
        </label>
      </div>

      {scenarioType === 'uniform_charge_change' && (
        <div className="scenario-builder-row">
          <label htmlFor="scenario-percent-change">
            Price Change: {percentChange > 0 ? '+' : ''}
            {percentChange}%
            <input
              id="scenario-percent-change"
              type="range"
              min="-50"
              max="50"
              step="1"
              value={percentChange}
              onChange={(event) => setPercentChange(Number(event.target.value))}
            />
          </label>
        </div>
      )}

      {scenarioType === 'contract_migration' && (
        <div className="scenario-builder-row">
          {contractOptions !== null && contractOptions.length < 2 && (
            <p className="error-text">
              Contract Migration isn't available for this tenant - it requires a resolved segment/contract-equivalent
              column, which needs Survival Analysis to have passed its own sanity gate first.
            </p>
          )}
          <label htmlFor="scenario-from-contract">
            From Contract
            <select
              id="scenario-from-contract"
              value={fromContract}
              onChange={(event) => setFromContract(event.target.value)}
            >
              {(contractOptions || []).map((opt) => (
                <option key={opt} value={opt}>
                  {opt}
                </option>
              ))}
            </select>
          </label>
          <label htmlFor="scenario-to-contract">
            To Contract
            <select id="scenario-to-contract" value={toContract} onChange={(event) => setToContract(event.target.value)}>
              {(contractOptions || []).map((opt) => (
                <option key={opt} value={opt}>
                  {opt}
                </option>
              ))}
            </select>
          </label>
          <label htmlFor="scenario-migration-rate">
            Migration Rate: {Math.round(migrationRate * 100)}%
            <input
              id="scenario-migration-rate"
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={migrationRate}
              onChange={(event) => setMigrationRate(Number(event.target.value))}
            />
          </label>
        </div>
      )}

      {scenarioType === 'discount_offer' && (
        <div className="scenario-builder-row">
          <label htmlFor="scenario-discount-percent">
            Discount: {discountPercent}%
            <input
              id="scenario-discount-percent"
              type="range"
              min="1"
              max="100"
              step="1"
              value={discountPercent}
              onChange={(event) => setDiscountPercent(Number(event.target.value))}
            />
          </label>
          <label htmlFor="scenario-target-segment">
            Target Segment
            <select
              id="scenario-target-segment"
              value={targetSegment}
              onChange={(event) => setTargetSegment(event.target.value)}
            >
              <option value="">All customers</option>
              {segments.map((segment) => (
                <option key={segment.cluster} value={segment.cluster}>
                  {segment.label}
                </option>
              ))}
            </select>
          </label>
        </div>
      )}

      {scenarioType === 'loyalty_program' && (
        <div className="scenario-builder-row">
          <label htmlFor="scenario-adoption-rate">
            Adoption Rate: {Math.round(adoptionRate * 100)}%
            <input
              id="scenario-adoption-rate"
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={adoptionRate}
              onChange={(event) => setAdoptionRate(Number(event.target.value))}
            />
          </label>
          <p className="muted small">
            Illustrative assumption: adopters get a flat 15% churn-probability reduction — a placeholder for
            demonstration, not a measured loyalty-program effect (no real outcome data exists in this system yet).
          </p>
        </div>
      )}

      <button type="submit" disabled={running}>
        {running ? 'Running Scenario…' : 'Run Scenario'}
      </button>
    </form>
  )
}
