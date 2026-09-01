import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'

export default function WelcomeScreen() {
  const { user, dismissJustRegistered } = useAuth()
  const companyName = user?.companyName || 'your'

  // Clear the one-time flag that routed here as soon as this screen is
  // shown, so returning to "/" later (e.g. the "Explore the Dashboard"
  // button below, which links to "/") renders the Dashboard instead of
  // redirecting back here again.
  useEffect(() => {
    dismissJustRegistered()
  }, [dismissJustRegistered])

  return (
    <div className="login-page">
      <div className="register-welcome">
        <h1>Welcome to {companyName}'s workspace</h1>
        <p className="muted">Your account is ready. Here's a good place to start:</p>
        <div className="register-welcome-actions">
          <Link to="/data-onboarding" className="register-welcome-button register-welcome-button-primary">
            Upload Your Data
          </Link>
          <Link to="/" className="register-welcome-button">
            Explore the Dashboard
          </Link>
          <Link to="/settings/api-keys" className="register-welcome-button">
            Manage API Keys
          </Link>
        </div>
      </div>
    </div>
  )
}
