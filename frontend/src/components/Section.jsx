export default function Section({ title, loading, error, notAvailable, children }) {
  return (
    <section className="dashboard-section">
      <h2>{title}</h2>
      {loading && <p className="muted">Loading…</p>}
      {!loading && error && <p className="error-text">Failed to load: {error}</p>}
      {!loading && !error && notAvailable && <p className="not-available">{notAvailable}</p>}
      {!loading && !error && !notAvailable && children}
    </section>
  )
}
