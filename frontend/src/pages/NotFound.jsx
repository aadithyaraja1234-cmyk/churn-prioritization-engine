import { Link } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'

// Both of App.jsx's previous catch-all routes silently redirected instead of
// showing anything - a mistyped/stale URL (e.g. an old bookmark to a page
// that's since moved) looked identical to a normal navigation, with no
// signal to the user that the thing they asked for doesn't exist. This is
// the one real 404 state, reused for both the authenticated and
// unauthenticated route trees (see App.jsx).
export default function NotFound() {
  const { isAuthenticated } = useAuth()

  return (
    <div className="landing-page">
      <div className="landing-card">
        <h1>Page not found</h1>
        <p>The page you're looking for doesn't exist, or the link may be out of date.</p>
        <div className="landing-actions">
          <Link className="landing-signin-button" to="/">
            {isAuthenticated ? 'Back to dashboard' : 'Back to sign in'}
          </Link>
        </div>
      </div>
    </div>
  )
}
