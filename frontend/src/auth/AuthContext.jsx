import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import client, { setAuthToken, setUnauthorizedHandler } from '../api/client'
import { useToast } from '../hooks/useToast'

const AuthContext = createContext(null)

function decodeJwtPayload(token) {
  const payload = token.split('.')[1]
  return JSON.parse(atob(payload))
}

export function AuthProvider({ children }) {
  // Token lives only in React state - never localStorage/sessionStorage/cookies.
  // A page refresh logs the user out; that's intentional for this session.
  const [token, setToken] = useState(null)
  const [user, setUser] = useState(null)
  const { showToast } = useToast()

  // Exposed via context as-is (AppHeader's "Log out" button wires this
  // straight to onClick, so it must stay a zero-arg function - a message
  // param here would silently receive the click SyntheticEvent instead).
  const logout = useCallback(() => {
    setToken(null)
    setUser(null)
    setAuthToken(null)
  }, [])

  // Only for the involuntary case (a real 401 from an expired/invalid
  // token) - explains why the user is suddenly back at the login screen,
  // instead of the silent bounce this used to be.
  const logoutWithMessage = useCallback(
    (message) => {
      showToast(message)
      logout()
    },
    [showToast, logout],
  )

  useEffect(() => {
    setUnauthorizedHandler(() => logoutWithMessage('Your session expired - please log in again.'))
  }, [logoutWithMessage])

  const login = useCallback(async (email, password) => {
    const response = await client.post('/auth/login', { email, password })
    const accessToken = response.data.access_token
    const claims = decodeJwtPayload(accessToken)
    setAuthToken(accessToken)
    setToken(accessToken)
    setUser({ email: claims.sub, tenantId: claims.tenant_id, role: claims.role })
  }, [])

  const registerCompany = useCallback(async (companyName, email, password) => {
    const response = await client.post('/api/companies/register', {
      company_name: companyName,
      email,
      password,
    })
    const { access_token: accessToken, tenant_id: tenantId, company_name: registeredCompanyName, onboarding_step: onboardingStep } =
      response.data
    const claims = decodeJwtPayload(accessToken)
    setAuthToken(accessToken)
    setToken(accessToken)
    // justRegistered drives the one-time welcome-screen redirect below (see
    // App.jsx's "/" route) - NOT an imperative navigate('/welcome') call from
    // RegisterPage, because RegisterPage unmounts the instant isAuthenticated
    // flips true (the route tree swaps out from under it), which raced with
    // and lost to the "/" catch-all before that navigate() could reliably
    // apply. Deriving the redirect from user state instead makes it a pure
    // function of the next render, immune to that unmount race.
    setUser({
      email: claims.sub,
      tenantId: claims.tenant_id,
      role: claims.role,
      companyName: registeredCompanyName,
      justRegistered: true,
    })
    return { tenantId, companyName: registeredCompanyName, onboardingStep }
  }, [])

  const dismissJustRegistered = useCallback(() => {
    setUser((prev) => (prev && prev.justRegistered ? { ...prev, justRegistered: false } : prev))
  }, [])

  const value = useMemo(
    () => ({ token, user, login, registerCompany, dismissJustRegistered, logout, isAuthenticated: Boolean(token) }),
    [token, user, login, registerCompany, dismissJustRegistered, logout],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return ctx
}
