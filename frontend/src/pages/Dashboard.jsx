import ActionQueue from '../components/ActionQueue'
import AppHeader from '../components/AppHeader'
import BusinessImpactKpiStrip from '../components/BusinessImpactKpiStrip'
import FeatureCoverage from '../components/FeatureCoverage'
import GettingStartedChecklist, { useOnboardingStatus } from '../components/GettingStartedChecklist'
import MorningBrief from '../components/MorningBrief'
import RecentAlerts from '../components/RecentAlerts'
import TopOpportunities from '../components/TopOpportunities'
import WelcomeIntro from '../components/WelcomeIntro'

export default function Dashboard() {
  const { status, checked } = useOnboardingStatus()
  // Onboarding isn't finished for this tenant - lead with the checklist and
  // skip the panels that would otherwise all just show "not yet trained for
  // this tenant" noise underneath it.
  const stillOnboarding = checked && status && status.step !== 'ready'

  return (
    <div className="dashboard">
      <AppHeader title="Churn Prioritization Dashboard" />

      {!checked && <p className="muted">Loading…</p>}

      {checked && (stillOnboarding ? (
        <GettingStartedChecklist status={status} />
      ) : (
        <>
          <MorningBrief />
          <WelcomeIntro />
          <FeatureCoverage />
          <BusinessImpactKpiStrip />
          <ActionQueue />
          <TopOpportunities />
          <RecentAlerts />
        </>
      ))}
    </div>
  )
}
