import AlertsPanel from '../components/AlertsPanel'
import AppHeader from '../components/AppHeader'
import CohortComparison from '../components/CohortComparison'
import FeatureImportanceChart from '../components/FeatureImportanceChart'
import KpiStrip from '../components/KpiStrip'
import SegmentCards from '../components/SegmentCards'

export default function AdminView() {
  return (
    <div className="dashboard">
      <AppHeader title="Admin — Model Internals" />

      <KpiStrip />
      <FeatureImportanceChart />
      <SegmentCards />
      <AlertsPanel />
      {/* Deliberately cross-tenant, unlike everything else on this page -
          /api/cohort-comparison always returns the same real Telco-vs-
          Banking aggregate comparison regardless of the caller's own
          tenant (see tests/test_cohort_comparison.py's
          test_cohort_comparison_accessible_regardless_of_caller_tenant).
          Not tenant-scoped by design, not by oversight - do not add
          gating here. */}
      <CohortComparison />
    </div>
  )
}
