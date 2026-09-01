import AnomalyCards from '../components/AnomalyCards'
import AppHeader from '../components/AppHeader'
import BacktestChart from '../components/BacktestChart'
import ClvSection from '../components/ClvSection'
import PriorityTable from '../components/PriorityTable'
import SurvivalChart from '../components/SurvivalChart'
import SurvivalLikelihoodPanel from '../components/SurvivalLikelihoodPanel'
import WhatIfPanel from '../components/WhatIfPanel'

// CohortComparison (Telco vs Banking) is deliberately not rendered here -
// it lives on the Admin page (see AdminView.jsx) instead, alongside the
// other cross-tenant/model-internals views, not this tenant-scoped
// Analytics page. Banking is validation-only evidence (models/banking_v1 +
// its test suite), not a live-demoed tenant - see docs/ADDING_A_TENANT.md.

export default function Analytics() {
  return (
    <div className="dashboard">
      <AppHeader title="Analytics" />

      <PriorityTable />
      <WhatIfPanel />
      <BacktestChart />
      <SurvivalLikelihoodPanel />
      <SurvivalChart />
      <AnomalyCards />
      <ClvSection />
    </div>
  )
}
