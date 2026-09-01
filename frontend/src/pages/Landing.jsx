import { Link } from 'react-router-dom'

export default function Landing() {
  return (
    <div className="landing-page">
      <div className="landing-card">
        <h1>Churn Prioritization Engine</h1>
        <p>See which customers are about to churn, why, and what to do about it — grounded in real model output.</p>
        <div className="landing-actions">
          <Link className="landing-signin-button" to="/login">
            Sign In
          </Link>
          <Link className="landing-register-link" to="/register">
            New company? Register your workspace
          </Link>
        </div>
        <p className="landing-register-link">
          <Link to="/privacy">Privacy Policy</Link> · <Link to="/terms">Terms of Service</Link>
        </p>
      </div>
    </div>
  )
}
