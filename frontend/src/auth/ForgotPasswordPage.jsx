import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useAuth } from './AuthContext'
import { emailError } from '../utils/credentialValidation'

export default function ForgotPasswordPage() {
  const { forgotPassword } = useAuth()
  const [email, setEmail] = useState('')
  const [touched, setTouched] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')

  async function handleSubmit(event) {
    event.preventDefault()
    setTouched(true)
    setError('')
    if (emailError(email)) return

    setSubmitting(true)
    try {
      const responseMessage = await forgotPassword(email)
      setMessage(responseMessage)
    } catch (err) {
      setError(err.response?.data?.detail || 'Something went wrong - please try again.')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="login-page">
      <form className="login-form" onSubmit={handleSubmit}>
        <h1>Reset your password</h1>
        {message ? (
          <>
            <p className="muted">{message}</p>
            <p className="register-login-link">
              <Link to="/login">Back to log in</Link>
            </p>
          </>
        ) : (
          <>
            <p className="muted">Enter your account email and we'll send you a link to reset your password.</p>
            <label>
              Email
              <input
                type="email"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                onBlur={() => setTouched(true)}
                autoComplete="username"
                required
              />
              {touched && emailError(email) && <span className="field-error">{emailError(email)}</span>}
            </label>
            {error && <p className="error-text">{error}</p>}
            <button type="submit" disabled={submitting}>
              {submitting ? 'Sending…' : 'Send reset link'}
            </button>
            <p className="register-login-link">
              <Link to="/login">Back to log in</Link>
            </p>
          </>
        )}
      </form>
    </div>
  )
}
