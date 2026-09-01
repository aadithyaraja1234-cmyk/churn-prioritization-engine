import { useEffect } from 'react'
import { Navigate, Route, BrowserRouter as Router, Routes } from 'react-router-dom'
import './App.css'
import { AuthProvider, useAuth } from './auth/AuthContext'
import LoginPage from './auth/LoginPage'
import RegisterPage from './auth/RegisterPage'
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

function AppRoutes() {
  const { isAuthenticated, user } = useAuth()
  usePageTitle(isAuthenticated)

  if (!isAuthenticated) {
    return (
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route path="/privacy" element={<PrivacyPolicy />} />
        <Route path="/terms" element={<TermsOfService />} />
        {/* Was `<Landing />` for every unmatched path, which made a typo'd
            or stale link look identical to a normal visit to "/" - a real
            404 makes a broken link visibly distinct instead of silently
            standing in for the homepage. */}
        <Route path="*" element={<NotFound />} />
      </Routes>
    )
  }

  return (
    <Routes>
      <Route path="/" element={user?.justRegistered ? <Navigate to="/welcome" replace /> : <Dashboard />} />
      <Route path="/analytics" element={<Analytics />} />
      <Route path="/executive" element={<ExecutiveView />} />
      <Route path="/scenarios" element={<ScenarioSimulator />} />
      <Route path="/budget-optimizer" element={<BudgetOptimizer />} />
      <Route path="/customer-360" element={<Customer360 />} />
      <Route path="/copilot" element={<CopilotChat />} />
      <Route path="/settings/api-keys" element={<ApiKeysSettings />} />
      <Route path="/data-onboarding" element={<DataOnboarding />} />
      <Route path="/welcome" element={<WelcomeScreen />} />
      <Route path="/admin" element={<AdminRoute isAdmin={user?.role === 'admin'} />} />
      <Route path="/privacy" element={<PrivacyPolicy />} />
      <Route path="/terms" element={<TermsOfService />} />
      {/* Was `<Navigate to="/" replace />`, which silently bounced a stale/
          mistyped URL back to the dashboard with no indication anything was
          wrong - a real 404 instead. */}
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
