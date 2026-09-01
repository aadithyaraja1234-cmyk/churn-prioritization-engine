import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useAuth } from './AuthContext'
import { companyNameError, emailError, passwordError, passwordStrengthScore } from '../utils/credentialValidation'

const STRENGTH_LABELS = ['Very weak', 'Weak', 'Fair', 'Strong', 'Very strong']

function FieldError({ message }) {
  if (!message) return null
  return <span className="field-error">{message}</span>
}

export default function RegisterPage() {
  const { registerCompany } = useAuth()

  const [companyName, setCompanyName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [touched, setTouched] = useState({})
  const [submitting, setSubmitting] = useState(false)
  const [serverErrors, setServerErrors] = useState({})
  const [genericError, setGenericError] = useState('')

  const clientErrors = {
    company_name: companyNameError(companyName),
    email: emailError(email),
    password: passwordError(password),
  }

  function markTouched(field) {
    setTouched((prev) => ({ ...prev, [field]: true }))
  }

  function fieldMessage(field) {
    return serverErrors[field] || (touched[field] ? clientErrors[field] : null)
  }

  async function handleSubmit(event) {
    event.preventDefault()
    setTouched({ company_name: true, email: true, password: true })
    setGenericError('')
    setServerErrors({})

    if (clientErrors.company_name || clientErrors.email || clientErrors.password) {
      return
    }

    setSubmitting(true)
    try {
      // registerCompany() sets the auth token, which immediately swaps
      // AppRoutes over to the authenticated route tree, unmounting this page
      // - so we don't navigate('/welcome') here ourselves (this component
      // won't survive long enough for that to reliably land before the "/"
      // catch-all redirect fires first). AppRoutes derives the "/welcome"
      // redirect from user.justRegistered instead, which is immune to that
      // unmount race - see App.jsx and AuthContext.registerCompany().
      await registerCompany(companyName, email, password)
    } catch (err) {
      const detail = err.response?.data?.detail
      if (detail?.errors) {
        setServerErrors(detail.errors)
      } else {
        setGenericError(typeof detail === 'string' ? detail : 'Registration failed - please try again.')
      }
    } finally {
      setSubmitting(false)
    }
  }

  const strengthScore = passwordStrengthScore(password)

  return (
    <div className="login-page">
      <form className="login-form register-form" onSubmit={handleSubmit}>
        <h1>Register Your Company</h1>

        <label>
          Company Name
          <input
            type="text"
            value={companyName}
            onChange={(event) => setCompanyName(event.target.value)}
            onBlur={() => markTouched('company_name')}
            autoComplete="organization"
            required
          />
          <FieldError message={fieldMessage('company_name')} />
        </label>

        <label>
          Email
          <input
            type="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            onBlur={() => markTouched('email')}
            autoComplete="username"
            required
          />
          <FieldError message={fieldMessage('email')} />
        </label>

        <label>
          Password
          <input
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            onBlur={() => markTouched('password')}
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
          <FieldError message={fieldMessage('password')} />
        </label>

        {genericError && <p className="error-text">{genericError}</p>}

        <button type="submit" disabled={submitting}>
          {submitting ? 'Creating your workspace…' : 'Create Workspace'}
        </button>

        <p className="register-login-link">
          Already have an account? <Link to="/login">Sign in</Link>
        </p>
        <p className="register-login-link">
          By creating a workspace you agree to our <Link to="/terms">Terms</Link> and{' '}
          <Link to="/privacy">Privacy Policy</Link>.
        </p>
      </form>
    </div>
  )
}
