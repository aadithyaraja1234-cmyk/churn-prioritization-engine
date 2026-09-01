import InterventionRateCaveat from './InterventionRateCaveat'

export default function BudgetCostCaveat({ result }) {
  const { metadata, cost_per_intervention: cost, cost_per_intervention_is_default: isDefault } = result

  return (
    <div className="intervention-caveat-block">
      <p className="caveat">
        Assumes ${cost.toFixed(0)} per intervention {isDefault ? '(default — not a measured cost)' : '(custom override)'} —{' '}
        {metadata.cost_per_intervention_note}
      </p>
      <InterventionRateCaveat metadata={metadata} label="Recoverable revenue for selected customers" />
    </div>
  )
}
