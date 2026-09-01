import { useState } from 'react'
import { useAuth } from '../auth/AuthContext'

function dismissalKey(email) {
  return `churn-engine:welcome-dismissed:${email}`
}

export default function WelcomeIntro() {
  const { user } = useAuth()
  const key = dismissalKey(user?.email || 'anonymous')
  const [dismissed, setDismissed] = useState(() => localStorage.getItem(key) === 'true')

  if (dismissed) return null

  function handleDismiss() {
    localStorage.setItem(key, 'true')
    setDismissed(true)
  }

  return (
    <section className="dashboard-section welcome-intro">
      <button type="button" className="welcome-intro-dismiss" onClick={handleDismiss} aria-label="Dismiss">
        Dismiss
      </button>
      <h2>Welcome</h2>
      <dl className="welcome-intro-terms">
        <dt>Revenue at Risk</dt>
        <dd>The total monthly revenue tied to customers this model flags as likely to churn.</dd>
        <dt>Opportunity Score</dt>
        <dd>A prioritization heuristic combining revenue, ease of saving, and customer value — highest first.</dd>
        <dt>Health Score</dt>
        <dd>A 0–100 rollup of a customer's churn risk and value, colored green/yellow/red.</dd>
      </dl>
    </section>
  )
}
