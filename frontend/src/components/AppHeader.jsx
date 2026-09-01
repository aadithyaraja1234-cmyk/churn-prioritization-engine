import { useEffect, useRef, useState } from 'react'
import { NavLink } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'

// Account-level/integration concerns (API Keys, Admin) - reachable via
// this menu rather than the primary nav, which is reserved for day-to-day
// product surfaces. Purely a navigation-structure change: every route
// below is unchanged, still fully functional, just re-categorized.
function AccountMenu({ isAdmin }) {
  const [open, setOpen] = useState(false)
  const menuRef = useRef(null)

  useEffect(() => {
    if (!open) return
    function handlePointerDown(event) {
      if (menuRef.current && !menuRef.current.contains(event.target)) setOpen(false)
    }
    function handleKeyDown(event) {
      if (event.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', handlePointerDown)
    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.removeEventListener('mousedown', handlePointerDown)
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [open])

  return (
    <div className="account-menu" ref={menuRef}>
      <button
        type="button"
        className="account-menu-trigger"
        onClick={() => setOpen((prev) => !prev)}
        aria-haspopup="true"
        aria-expanded={open}
        title="Settings"
      >
        ⚙️
      </button>
      {open && (
        <div className="account-menu-dropdown">
          <NavLink to="/settings/api-keys" onClick={() => setOpen(false)}>
            API Keys
          </NavLink>
          {isAdmin && (
            <NavLink to="/admin" onClick={() => setOpen(false)}>
              Admin
            </NavLink>
          )}
        </div>
      )}
    </div>
  )
}

export default function AppHeader({ title }) {
  const { user, logout } = useAuth()

  return (
    <header className="dashboard-header">
      <div className="dashboard-header-left">
        <h1>{title}</h1>
        <nav className="dashboard-nav">
          <NavLink to="/" end>Dashboard</NavLink>
          <NavLink to="/analytics">Analytics</NavLink>
          <NavLink to="/executive">Executive</NavLink>
          <NavLink to="/scenarios">Scenarios</NavLink>
          <NavLink to="/budget-optimizer">Budget Optimizer</NavLink>
          <NavLink to="/customer-360">Customer 360</NavLink>
          <NavLink to="/copilot">Copilot</NavLink>
          <NavLink to="/data-onboarding">Add Data</NavLink>
        </nav>
      </div>
      <div className="dashboard-header-right">
        <span>
          {user?.email} ({user?.tenantId})
        </span>
        <AccountMenu isAdmin={user?.role === 'admin'} />
        <button type="button" onClick={logout}>
          Log out
        </button>
      </div>
    </header>
  )
}
