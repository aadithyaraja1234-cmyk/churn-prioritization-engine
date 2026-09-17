import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import client, { setAuthToken, setUnauthorizedHandler } from '../api/client'
import { useToast } from '../hooks/useToast'

const AuthContext = createContext(null)

// Persisted in localStorage (not sessionStorage/a cookie) so a page refresh
// - or coming back tomorrow - doesn't log the user out. This is a deliberate
// reversal of an earlier decision here to keep the token in React state
// only: storing a JWT in localStorage is readable by any script that runs
// on this page (an XSS bug elsewhere in the app could steal it, unlike a
// backend-set httpOnly cookie), but for this app's risk profile the
// friendlier "stay logged in" UX was judged worth that tradeoff - the
// alternative (an httpOnly-cookie + refresh-token rotation flow) is the
// more secure answer but meaningfully more infrastructure for what's meant
// to stay a simple flow. See auth/auth.py's ACCESS_TOKEN_EXPIRE_MINUTES for
// the matching server-side expiry this now needs to actually be long-lived.
const TOKEN_STORAGE_KEY = 'churn_engine_token'

function decodeJwtPayload(token) {
  const payload = token.split('.')[1]
  return JSON.parse(atob(payload))
}

function loadPersistedAuth() {
  let storedToken
  try {
    storedToken = localStorage.getItem(TOKEN_STORAGE_KEY)
  } catch {
    // localStorage can throw (private-browsing mode in some browsers,
    // storage disabled by policy) - never let that crash the app; just
    // start logged out, same as if nothing were stored.
    return { token: null, user: null }
  }
  if (!storedToken) return { token: null, user: null }

  try {
    const claims = decodeJwtPayload(storedToken)
    if (claims.exp && claims.exp * 1000 < Date.now()) {
      localStorage.removeItem(TOKEN_STORAGE_KEY)
      return { token: null, user: null }
    }
    return {
      token: storedToken,
      user: { email: claims.sub, tenantId: claims.tenant_id, role: claims.role },
    }
  } catch {
    // Malformed/corrupted stored value (e.g. hand-edited in devtools) -
    // don't let a bad stored token crash the app on load.
    localStorage.removeItem(TOKEN_STORAGE_KEY)
    return { token: null, user: null }
  }
}

function persistToken(token) {
  try {
    if (token) {
      localStorage.setItem(TOKEN_STORAGE_KEY, token)
    } else {
      localStorage.removeItem(TOKEN_STORAGE_KEY)
    }
  } catch {
    // Storage full/disabled - the session still works for this tab via
    // React state, it just won't survive a refresh. Not worth surfacing
    // to the user over.
  }
}

export function AuthProvider({ children }) {
  const [{ token: initialToken, user: initialUser }] = useState(loadPersistedAuth)
  const [token, setToken] = useState(initialToken)
  const [user, setUser] = useState(initialUser)
  const { showToast } = useToast()

  // Restores the shared axios client's Authorization header on first
  // mount for a persisted session (login()/registerCompany() set it
  // directly for a fresh one) - without this, a restored token would sit
  // in state but never actually get sent on any request.
  useEffect(() => {
    if (initialToken) setAuthToken(initialToken)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Exposed via context as-is (AppHeader's "Log out" button wires this
  // straight to onClick, so it must stay a zero-arg function - a message
  // param here would silently receive the click SyntheticEvent instead).
  const logout = useCallback(() => {
    setToken(null)
    setUser(null)
    setAuthToken(null)
    persistToken(null)
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
    persistToken(accessToken)
    setUser({ email: claims.sub, tenantId: claims.tenant_id, role: claims.role })
  }, [])

  const forgotPassword = useCallback(async (email) => {
    const response = await client.post('/auth/forgot-password', { email })
    return response.data.message
  }, [])

  const resetPassword = useCallback(async (resetToken, newPassword) => {
    const response = await client.post('/auth/reset-password', { token: resetToken, new_password: newPassword })
    const accessToken = response.data.access_token
    const claims = decodeJwtPayload(accessToken)
    setAuthToken(accessToken)
    setToken(accessToken)
    persistToken(accessToken)
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
    persistToken(accessToken)
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
    () => ({
      token,
      user,
      login,
      registerCompany,
      forgotPassword,
      resetPassword,
      dismissJustRegistered,
      logout,
      isAuthenticated: Boolean(token),
    }),
    [token, user, login, registerCompany, forgotPassword, resetPassword, dismissJustRegistered, logout],
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
