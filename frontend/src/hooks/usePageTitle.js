import { useEffect } from 'react'
import { useLocation } from 'react-router-dom'

const BRAND = 'Churn Prioritization Engine'

// Centralized route -> title map, rather than scattering document.title
// calls across every page component - one place to keep in sync with
// App.jsx's route list. Matched by exact pathname; a route not listed here
// (there shouldn't be any real one - NotFound covers the true catch-all)
// just falls back to the bare brand name.
const ROUTE_TITLES = {
  '/': 'Dashboard',
  '/login': 'Sign In',
  '/register': 'Register Your Company',
  '/analytics': 'Analytics',
  '/executive': 'Executive View',
  '/scenarios': 'Scenario Simulator',
  '/budget-optimizer': 'Budget Optimizer',
  '/customer-360': 'Customer 360',
  '/copilot': 'Copilot',
  '/settings/api-keys': 'API Keys',
  '/data-onboarding': 'Data Onboarding',
  '/welcome': 'Welcome',
  '/admin': 'Admin',
  '/privacy': 'Privacy Policy',
  '/terms': 'Terms of Service',
}

export default function usePageTitle(isAuthenticated) {
  const location = useLocation()

  useEffect(() => {
    // "/" means two different real pages depending on auth state (Landing
    // vs Dashboard) - the route map alone can't distinguish them.
    const label = location.pathname === '/' && !isAuthenticated ? null : ROUTE_TITLES[location.pathname]
    document.title = label ? `${label} · ${BRAND}` : BRAND
  }, [location.pathname, isAuthenticated])
}
