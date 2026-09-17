import { useEffect } from 'react'
import { Navigate, Route, BrowserRouter as Router, Routes } from 'react-router-dom'
import './App.css'
import { AuthProvider, useAuth } from './auth/AuthContext'
import ForgotPasswordPage from './auth/ForgotPasswordPage'
import LoginPage from './auth/LoginPage'
import RegisterPage from './auth/RegisterPage'
import ResetPasswordPage from './auth/ResetPasswordPage'
import { ToastProvider, useToast } from './hooks/useToast'
import usePageTitle from './hooks/usePageTitle'
import ToastStack from './components/ToastStack'
import AdminView from './pages/AdminView'
import Analytics from './pages/Analytics'
import ApiKeysSettings from './pages/ApiKeysSettings'
import BudgetOptimizer from './pages/BudgetOptimizer'
import CopilotChat from './pages/CopilotChat'
import Customer360 from './pages/Customer360'
import Dashboard from './pages/Dashboard'
import DataOnboarding from './pages/DataOnboarding'
import WelcomeScreen from './pages/WelcomeScreen'
import ExecutiveView from './pages/ExecutiveView'
import Landing from './pages/Landing'
import NotFound from './pages/NotFound'
import PrivacyPolicy from './pages/PrivacyPolicy'
import ScenarioSimulator from './pages/ScenarioSimulator'
import TermsOfService from './pages/TermsOfService'

// A non-admin landing here is a client-side authorization check, not a
// backend 403 (the API has no admin-only endpoints to reject) - still
// worth an honest toast rather than a silent bounce back to "/", same
// discipline as the 401 case in AuthContext.
function AdminRoute({ isAdmin }) {
  const { showToast } = useToast()

  useEffect(() => {
    if (!isAdmin) showToast("You don't have access to this page.")
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAdmin])

  if (!isAdmin) return <Navigate to="/" replace />
  return <AdminView />
}

// Login/register succeeding (or logging out) flips isAuthenticated without
// ever calling navigate() - the browser URL doesn't change on its own. This
// used to be two entirely separate <Routes> trees swapped on isAuthenticated
// (logged-out-only paths like "/login"/"/register", logged-in-only paths
// like "/welcome"/"/analytics"), so flipping auth state while sitting on a
// path that only existed in the OTHER tree fell through to the "*" wildcard
// - e.g. logging in while still at "/login" landed on NotFound instead of
// the dashboard. One shared tree with every path always registered, gated
// per-route via RequireAuth/RequireGuest below, means the current path is
// always matched by something real regardless of which side of the auth
// flip you're on.
function RequireAuth({ isAuthenticated, children }) {
  if (!isAuthenticated) return <Navigate to="/login" replace />
  return children
}

function RequireGuest({ isAuthenticated, children }) {
  if (isAuthenticated) return <Navigate to="/" replace />
  return children
}

function AppRoutes() {
  const { isAuthenticated, user } = useAuth()
  usePageTitle(isAuthenticated)

  return (
    <Routes>
      <Route
        path="/"
        element={
          !isAuthenticated ? (
            <Landing />
          ) : user?.justRegistered ? (
            <Navigate to="/welcome" replace />
          ) : (
            <Dashboard />
          )
        }
      />
      <Route
        path="/login"
        element={
          <RequireGuest isAuthenticated={isAuthenticated}>
            <LoginPage />
          </RequireGuest>
        }
      />
      <Route
        path="/register"
        element={
          <RequireGuest isAuthenticated={isAuthenticated}>
            <RegisterPage />
          </RequireGuest>
        }
      />
      <Route
        path="/forgot-password"
        element={
          <RequireGuest isAuthenticated={isAuthenticated}>
            <ForgotPasswordPage />
          </RequireGuest>
        }
      />
      <Route
        path="/reset-password"
        element={
          <RequireGuest isAuthenticated={isAuthenticated}>
            <ResetPasswordPage />
          </RequireGuest>
        }
      />
      <Route
        path="/analytics"
        element={
          <RequireAuth isAuthenticated={isAuthenticated}>
            <Analytics />
          </RequireAuth>
        }
      />
      <Route
        path="/executive"
        element={
          <RequireAuth isAuthenticated={isAuthenticated}>
            <ExecutiveView />
          </RequireAuth>
        }
      />
      <Route
        path="/scenarios"
        element={
          <RequireAuth isAuthenticated={isAuthenticated}>
            <ScenarioSimulator />
          </RequireAuth>
        }
      />
      <Route
        path="/budget-optimizer"
        element={
          <RequireAuth isAuthenticated={isAuthenticated}>
            <BudgetOptimizer />
          </RequireAuth>
        }
      />
      <Route
        path="/customer-360"
        element={
          <RequireAuth isAuthenticated={isAuthenticated}>
            <Customer360 />
          </RequireAuth>
        }
      />
      <Route
        path="/copilot"
        element={
          <RequireAuth isAuthenticated={isAuthenticated}>
            <CopilotChat />
          </RequireAuth>
        }
      />
      <Route
        path="/settings/api-keys"
        element={
          <RequireAuth isAuthenticated={isAuthenticated}>
            <ApiKeysSettings />
          </RequireAuth>
        }
      />
      <Route
        path="/data-onboarding"
        element={
          <RequireAuth isAuthenticated={isAuthenticated}>
            <DataOnboarding />
          </RequireAuth>
        }
      />
      <Route
        path="/welcome"
        element={
          <RequireAuth isAuthenticated={isAuthenticated}>
            <WelcomeScreen />
          </RequireAuth>
        }
      />
      <Route
        path="/admin"
        element={
          <RequireAuth isAuthenticated={isAuthenticated}>
            <AdminRoute isAdmin={user?.role === 'admin'} />
          </RequireAuth>
        }
      />
      <Route path="/privacy" element={<PrivacyPolicy />} />
      <Route path="/terms" element={<TermsOfService />} />
      {/* Was `<Landing />`/`<Navigate to="/" replace />` for every unmatched
          path (depending on which of the two old route trees was active),
          which made a typo'd or stale link look identical to a normal visit
          - a real 404 makes a broken link visibly distinct instead of
          silently standing in for the homepage/dashboard. */}
      <Route path="*" element={<NotFound />} />
    </Routes>
  )
}

export default function App() {
  return (
    <ToastProvider>
      <AuthProvider>
        <Router>
          <AppRoutes />
        </Router>
        <ToastStack />
      </AuthProvider>
    </ToastProvider>
  )
}
