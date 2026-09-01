import { useEffect, useState } from 'react'
import client from '../api/client'
import { useToast } from '../hooks/useToast'

const STATUS_OPTIONS = ['not_started', 'called', 'emailed', 'resolved']

export default function TrackingFields({ customerId }) {
  const [tracking, setTracking] = useState(null)
  const [saving, setSaving] = useState(false)
  const { showToast } = useToast()

  useEffect(() => {
    let cancelled = false
    client.get(`/api/tracking/${customerId}`).then((response) => {
      if (!cancelled) setTracking(response.data)
    })
    return () => {
      cancelled = true
    }
  }, [customerId])

  async function save(patch) {
    const next = { ...tracking, ...patch }
    setTracking(next)
    setSaving(true)
    try {
      const response = await client.put(`/api/tracking/${customerId}`, {
        assigned_manager: next.assigned_manager || null,
        status: next.status,
        call_scheduled_date: next.call_scheduled_date || null,
      })
      setTracking(response.data)
      showToast('Tracking updated')
    } finally {
      setSaving(false)
    }
  }

  if (!tracking) return <span className="muted small">…</span>

  return (
    <div className="tracking-fields">
      <input
        type="text"
        className="tracking-manager-input"
        placeholder="Unassigned"
        value={tracking.assigned_manager || ''}
        onChange={(event) => setTracking({ ...tracking, assigned_manager: event.target.value })}
        onBlur={(event) => save({ assigned_manager: event.target.value })}
        disabled={saving}
      />
      <select
        className="tracking-status-select"
        value={tracking.status}
        onChange={(event) => save({ status: event.target.value })}
        disabled={saving}
      >
        {STATUS_OPTIONS.map((status) => (
          <option key={status} value={status}>
            {status.replaceAll('_', ' ')}
          </option>
        ))}
      </select>
      <input
        type="date"
        className="tracking-date-input"
        value={tracking.call_scheduled_date ? tracking.call_scheduled_date.slice(0, 10) : ''}
        onChange={(event) => save({ call_scheduled_date: event.target.value || null })}
        disabled={saving}
      />
    </div>
  )
}
