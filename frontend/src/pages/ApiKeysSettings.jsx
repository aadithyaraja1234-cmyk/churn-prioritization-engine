import { useCallback, useEffect, useState } from 'react'
import client from '../api/client'
import AppHeader from '../components/AppHeader'
import Section from '../components/Section'
import { useToast } from '../hooks/useToast'

export default function ApiKeysSettings() {
  const { showToast } = useToast()
  const [keys, setKeys] = useState(null)
  const [error, setError] = useState(null)

  const [name, setName] = useState('')
  const [creating, setCreating] = useState(false)
  const [createError, setCreateError] = useState(null)
  const [revealedKey, setRevealedKey] = useState(null)
  const [copied, setCopied] = useState(false)

  const fetchKeys = useCallback(() => {
    client
      .get('/api/api-keys')
      .then((response) => setKeys(response.data))
      .catch((err) => setError(err.response?.data?.detail || err.message))
  }, [])

  useEffect(() => {
    fetchKeys()
  }, [fetchKeys])

  async function handleCreate(event) {
    event.preventDefault()
    if (!name.trim()) return
    setCreating(true)
    setCreateError(null)
    setCopied(false)
    try {
      const response = await client.post('/api/api-keys', { name: name.trim() })
      setRevealedKey(response.data)
      setName('')
      fetchKeys()
    } catch (err) {
      setCreateError(err.response?.data?.detail || err.message)
    } finally {
      setCreating(false)
    }
  }

  async function handleCopy() {
    if (!revealedKey) return
    try {
      await navigator.clipboard.writeText(revealedKey.api_key)
      setCopied(true)
      showToast('API key copied to clipboard')
    } catch {
      setCopied(false)
    }
  }

  async function handleRevoke(id) {
    if (!window.confirm('Revoke this API key? Any integration using it will stop working immediately.')) {
      return
    }
    try {
      await client.delete(`/api/api-keys/${id}`)
      showToast('API key revoked')
      fetchKeys()
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    }
  }

  return (
    <div className="dashboard">
      <AppHeader title="API Keys" />

      <section className="dashboard-section">
        <h2>Create a New API Key</h2>
        <p className="muted">
          API keys let a system integration call a subset of read/scenario endpoints directly,
          without logging in as a user. Each key is scoped to your tenant only.
        </p>
        <form className="api-key-create-form" onSubmit={handleCreate}>
          <input
            type="text"
            placeholder="Key name (e.g. billing-integration)"
            value={name}
            onChange={(event) => setName(event.target.value)}
            disabled={creating}
          />
          <button type="submit" disabled={creating || !name.trim()}>
            {creating ? 'Generating…' : 'Generate Key'}
          </button>
        </form>
        {createError && <p className="error-text">{createError}</p>}

        {revealedKey && (
          <div className="api-key-reveal">
            <p className="api-key-reveal-warning">
              <strong>Copy this key now.</strong> For security, it will never be shown again after
              you leave this page.
            </p>
            <div className="api-key-reveal-row">
              <code className="api-key-reveal-value">{revealedKey.api_key}</code>
              <button type="button" onClick={handleCopy}>
                {copied ? 'Copied!' : 'Copy'}
              </button>
            </div>
          </div>
        )}
      </section>

      <Section title="Your API Keys" loading={!keys && !error} error={error}>
        {keys && keys.length === 0 && (
          <p className="muted">No API keys yet — create one above to get started.</p>
        )}
        {keys && keys.length > 0 && (
          <table className="api-keys-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Key</th>
                <th>Rate Limit</th>
                <th>Created</th>
                <th>Last Used</th>
                <th>Status</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {keys.map((key) => (
                <tr key={key.id} className={key.revoked ? 'api-key-row-revoked' : ''}>
                  <td>{key.name}</td>
                  <td>
                    <code>{key.masked_key}</code>
                  </td>
                  <td>{key.rate_limit_per_minute}/min</td>
                  <td>{new Date(key.created_at).toLocaleString()}</td>
                  <td>{key.last_used_at ? new Date(key.last_used_at).toLocaleString() : 'Never'}</td>
                  <td>
                    {key.revoked ? (
                      <span className="api-key-status-revoked">Revoked</span>
                    ) : (
                      <span className="api-key-status-active">Active</span>
                    )}
                  </td>
                  <td>
                    {!key.revoked && (
                      <button type="button" className="api-key-revoke-button" onClick={() => handleRevoke(key.id)}>
                        Revoke
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Section>
    </div>
  )
}
