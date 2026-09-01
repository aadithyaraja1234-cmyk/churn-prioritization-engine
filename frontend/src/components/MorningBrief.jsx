import { useEffect, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import client from '../api/client'
import { MARKDOWN_COMPONENTS } from '../utils/markdownComponents'

export default function MorningBrief() {
  const [brief, setBrief] = useState(null)
  const [loading, setLoading] = useState(true)
  const [unavailable, setUnavailable] = useState(false)

  useEffect(() => {
    let cancelled = false
    client
      .get('/api/copilot/morning-brief')
      .then((response) => {
        if (!cancelled) setBrief(response.data)
      })
      .catch(() => {
        // Proactive, unprompted feature - a failure here should read as
        // "not available right now," not as a broken app on first login.
        if (!cancelled) setUnavailable(true)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  if (unavailable) return null

  return (
    <section className="dashboard-section morning-brief">
      <h2>Morning Brief</h2>
      {loading && <p className="muted">Generating your morning brief from real, current data…</p>}
      {brief && (
        <>
          <div className="copilot-markdown">
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={MARKDOWN_COMPONENTS}>
              {brief.response}
            </ReactMarkdown>
          </div>
          <p className="copilot-sources">
            Sources: {brief.sources.length > 0 ? brief.sources.join(', ') : 'none (no tool was needed)'}
          </p>
        </>
      )}
    </section>
  )
}
