import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import client from '../api/client'

const CHECKLIST_STEPS = [
  { key: 'registered', label: 'Create your workspace' },
  { key: 'data_uploaded', label: 'Upload your data' },
  { key: 'data_validated', label: 'Review validation results' },
  { key: 'ready', label: 'Data ready for use' },
]

const STEP_ORDER = CHECKLIST_STEPS.map((s) => s.key)

// Returns null while loading/unavailable, or { step, isReady } once known.
export function useOnboardingStatus() {
  const [status, setStatus] = useState(null)
  const [checked, setChecked] = useState(false)

  useEffect(() => {
    let cancelled = false
    client
      .get('/api/companies/onboarding-status')
      .then((response) => {
        if (!cancelled) setStatus(response.data)
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setChecked(true)
      })
    return () => {
      cancelled = true
    }
  }, [])

  return { status, checked }
}

export default function GettingStartedChecklist({ status }) {
  if (!status || status.step === 'ready') return null

  const currentIndex = STEP_ORDER.indexOf(status.step)

  return (
    <section className="getting-started-card">
      <h2>Getting Started</h2>
      <p className="muted">Finish setting up your workspace to unlock the full dashboard.</p>
      <ul className="getting-started-list">
        {CHECKLIST_STEPS.map((step, index) => {
          const done = index < currentIndex
          const current = index === currentIndex
          return (
            <li
              key={step.key}
              className={`getting-started-item ${done ? 'getting-started-item-done' : ''} ${
                current ? 'getting-started-item-current' : ''
              }`}
            >
              <span className="getting-started-checkbox">{done ? '✓' : '○'}</span>
              <span>{step.label}</span>
            </li>
          )
        })}
      </ul>
      <Link to="/data-onboarding" className="getting-started-cta">
        {status.step === 'registered' ? 'Upload Your Data' : 'Continue Onboarding'}
      </Link>
    </section>
  )
}
