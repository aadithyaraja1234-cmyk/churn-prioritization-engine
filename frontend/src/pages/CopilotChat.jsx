import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import client from '../api/client'
import AppHeader from '../components/AppHeader'
import { MARKDOWN_COMPONENTS } from '../utils/markdownComponents'

let nextMessageId = 1

export default function CopilotChat() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [unavailableError, setUnavailableError] = useState(null)

  async function handleSubmit(event) {
    event.preventDefault()
    const question = input.trim()
    if (!question || sending) return

    setInput('')
    setUnavailableError(null)
    setMessages((prev) => [...prev, { id: nextMessageId++, role: 'user', text: question }])
    setSending(true)

    try {
      const response = await client.post('/api/copilot/chat', { message: question })
      setMessages((prev) => [
        ...prev,
        {
          id: nextMessageId++,
          role: 'assistant',
          text: response.data.response,
          sources: response.data.sources || [],
        },
      ])
    } catch (err) {
      if (err.response?.status === 503) {
        setUnavailableError(err.response.data?.detail || 'Copilot is unavailable.')
      } else {
        setUnavailableError(err.response?.data?.detail || err.message)
      }
    } finally {
      setSending(false)
    }
  }

  return (
    <div className="dashboard">
      <AppHeader title="AI Copilot" />

      <section className="dashboard-section copilot-section">
        <p className="muted small">
          Answers are grounded only in this system's real tools (business impact, opportunities, customer lookups,
          budget optimization, scenarios, segments, alerts, survival likelihood) — it will tell you explicitly when
          something can't be answered rather than guessing.
        </p>

        <div className="copilot-messages">
          {messages.length === 0 && !sending && (
            <p className="muted">
              Ask something like "What is our total revenue at risk?" or "Who are our top opportunities?"
            </p>
          )}
          {messages.map((message) => (
            <div
              key={message.id}
              className={`copilot-message ${message.role === 'user' ? 'copilot-message-user' : 'copilot-message-assistant'}`}
            >
              <div className="copilot-message-bubble">
                {message.role === 'assistant' ? (
                  <div className="copilot-markdown">
                    <ReactMarkdown remarkPlugins={[remarkGfm]} components={MARKDOWN_COMPONENTS}>
                      {message.text}
                    </ReactMarkdown>
                  </div>
                ) : (
                  <p>{message.text}</p>
                )}
                {message.role === 'assistant' && (
                  <p className="copilot-sources">
                    Sources: {message.sources.length > 0 ? message.sources.join(', ') : 'none (no tool was needed)'}
                  </p>
                )}
              </div>
            </div>
          ))}
          {sending && (
            <div className="copilot-message copilot-message-assistant">
              <div className="copilot-message-bubble">
                <p className="muted">Thinking…</p>
              </div>
            </div>
          )}
        </div>

        {unavailableError && <p className="error-text">Copilot request failed: {unavailableError}</p>}

        <form className="copilot-input-row" onSubmit={handleSubmit}>
          <input
            type="text"
            placeholder="Ask the copilot a question…"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            disabled={sending}
          />
          <button type="submit" disabled={sending || !input.trim()}>
            {sending ? 'Sending…' : 'Send'}
          </button>
        </form>
      </section>
    </div>
  )
}
