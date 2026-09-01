export default function CoverageBar({ pct }) {
  const clamped = Math.max(0, Math.min(1, pct))

  return (
    <div className="coverage-bar-block">
      <div className="coverage-bar-label">
        <span>Portfolio Revenue at Risk covered at this budget</span>
        <span>{(clamped * 100).toFixed(1)}%</span>
      </div>
      <div className="coverage-bar-track">
        <div className="coverage-bar-fill" style={{ width: `${clamped * 100}%` }} />
      </div>
    </div>
  )
}
