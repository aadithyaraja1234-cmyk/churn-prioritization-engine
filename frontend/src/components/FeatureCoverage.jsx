import { useApiData } from '../hooks/useApiData'
import Section from './Section'

// No hardcoded per-tenant text anywhere here - every label/enabled/reason
// value comes straight from GET /api/tenant-features, which itself reads
// live from src.tenant_registry's feature_enabled()/unavailable_response()
// (the same source of truth every gated endpoint already uses).
export default function FeatureCoverage() {
  const { data, loading, error } = useApiData('/api/tenant-features')
  const rawFeatures = data?.features
  // clv_estimated is mutually exclusive with clv by design (see
  // src/models/tenant_training.py's _run_optional_modules()) - a tenant
  // with real CLV enabled will ALWAYS show clv_estimated as disabled, not
  // because anything is missing, but because the real model already
  // covers it. Showing that crossed-out badge next to the real one reads
  // as "something's broken/incomplete" rather than "you have the better
  // version" - so it's hidden here specifically when clv is enabled,
  // instead of adding a third enabled/disabled/not-applicable state to
  // GET /api/tenant-features just for this one pairing.
  const clvEnabled = rawFeatures?.some((f) => f.key === 'clv' && f.enabled)
  const features = rawFeatures?.filter((f) => !(f.key === 'clv_estimated' && clvEnabled))

  return (
    <Section title="Available for your account" loading={loading} error={error} notAvailable={null}>
      {features && (
        <div className="feature-coverage-list">
          {features.map((feature) => (
            <span
              key={feature.key}
              className={`feature-badge ${feature.enabled ? 'feature-badge-enabled' : 'feature-badge-disabled'}`}
              title={feature.enabled ? undefined : feature.reason}
            >
              {feature.enabled ? '✓' : '✗'} {feature.label}
            </span>
          ))}
        </div>
      )}
      {features && features.some((f) => !f.enabled) && (
        <p className="muted small feature-coverage-note">
          Greyed-out features aren't available yet — hover one to see why (usually: not yet trained/validated for
          your tenant).
        </p>
      )}
    </Section>
  )
}
