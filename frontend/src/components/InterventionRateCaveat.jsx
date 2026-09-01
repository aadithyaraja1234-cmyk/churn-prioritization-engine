export default function InterventionRateCaveat({ metadata, label }) {
  const hillstrom = metadata.hillstrom_benchmark

  return (
    <div className="intervention-caveat-block">
      <p className="caveat">
        {label} assumes a {(metadata.intervention_success_rate * 100).toFixed(0)}% intervention success rate —{' '}
        {metadata.intervention_success_rate_note}
      </p>
      {hillstrom && (
        <p className="caveat caveat-benchmark">
          Real-world reference point (cited evidence, not validation of the number above): the Hillstrom (2008)
          email-marketing RCT measured a real conversion-rate lift of{' '}
          {(hillstrom.conversion_absolute_uplift * 100).toFixed(2)} percentage points (
          {(hillstrom.conversion_relative_uplift * 100).toFixed(0)}% relative) from a targeted email vs. no email —
          a different domain (retail purchase) and base rate than this system's churn-retention use case, so it
          isn't a direct measurement of the {(metadata.intervention_success_rate * 100).toFixed(0)}% figure above.
        </p>
      )}
    </div>
  )
}
