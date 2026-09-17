import { useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { useAuth } from './AuthContext'
import { passwordError, passwordStrengthScore } from '../utils/credentialValidation'

const STRENGTH_LABELS = ['Very weak', 'Weak', 'Fair', 'Strong', 'Very strong']

export default function ResetPasswordPage() {
  const { resetPassword } = useAuth()
  const [searchParams] = useSearchParams()
  const token = searchParams.get('token') || ''

  const [password, setPassword] = useState('')
  const [touched, setTouched] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')

  const strengthScore = passwordStrengthScore(password)

  async function handleSubmit(event) {
    event.preventDefault()
    setTouched(true)
    setError('')
    if (passwordError(password)) return

    setSubmitting(true)
    try {
      // On success this flips isAuthenticated true - App.jsx's route
      // guard for "/reset-password" then redirects to the dashboard on
      // its own re-render, no imperative navigate() needed here.
      await resetPassword(token, password)
    } catch (err) {
      setError(err.response?.data?.detail || 'This reset link is invalid or has expired - request a new one.')
    } finally {
      setSubmitting(false)
    }
  }

  if (!token) {
    return (
      <div className="login-page">
        <div className="login-form">
          <h1>Reset your password</h1>
          <p className="error-text">This link is missing its reset token - use the link from your email.</p>
          <p className="register-login-link">
            <Link to="/forgot-password">Request a new reset link</Link>
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="login-page">
      <form className="login-form" onSubmit={handleSubmit}>
        <h1>Choose a new password</h1>
        <label>
          New password
          <input
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            onBlur={() => setTouched(true)}
            autoComplete="new-password"
            required
          />
          {password && (
            <div className="password-strength-meter" aria-hidden="true">
              <div className={`password-strength-bar password-strength-${strengthScore}`}>
                {[0, 1, 2, 3].map((segment) => (
                  <span key={segment} className={segment < strengthScore ? 'filled' : ''} />
                ))}
              </div>
              <span className="password-strength-label">{STRENGTH_LABELS[strengthScore]}</span>
            </div>
          )}
          {touched && passwordError(password) && <span className="field-error">{passwordError(password)}</span>}
        </label>
        {error && <p className="error-text">{error}</p>}
        <button type="submit" disabled={submitting}>
          {submitting ? 'Resetting…' : 'Reset password'}
        </button>
        <p className="register-login-link">
          <Link to="/login">Back to log in</Link>
        </p>
      </form>
    </div>
  )
}
