import { useEffect, useMemo, useState } from 'react'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import client from '../api/client'
import { CHART_AXIS_TICK, CHART_GRID_STROKE, CHART_LEGEND_WRAPPER_STYLE, CHART_TOOLTIP_PROPS } from '../styles/chartTheme'
import Section from './Section'

const DEFAULT_PCT = 20
const VALIDATED_CHECKPOINTS = [10, 20, 30]

// CVD-safe (blue/amber/gray - avoids the red/green pairs deuteranopia/
// protanopia most commonly confuse) - deliberately NOT drawn from
// tokens.css's accent/success/danger palette, which is violet/green/red
// and would reintroduce exactly that confusable pairing across 3 series.
// Verified >=3:1 against --color-surface: blue 3.48:1, amber 6.12:1,
// gray 7.08:1 - left unchanged from the original fix.
const STRATEGY_COLORS = {
  revenue_weighted: '#2563eb',
  probability_only: '#ca8a04',
  random: '#9ca3af',
}
const STRATEGY_LABELS = {
  revenue_weighted: 'Revenue-Weighted',
  probability_only: 'Probability-Only',
  random: 'Random',
}

function formatCurrency(n) {
  return `$${n.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
}

function liftClass(pointsDiff) {
  if (pointsDiff > 0.5) return 'roi-lift-positive'
  if (pointsDiff < -0.5) return 'roi-lift-negative'
  return 'roi-lift-flat'
}

function formatLift(pointsDiff) {
  const sign = pointsDiff > 0 ? '+' : ''
  return `${sign}${pointsDiff.toFixed(1)}pp`
}

export default function BacktestChart() {
  const [curve, setCurve] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [notAvailable, setNotAvailable] = useState(null)
  const [pct, setPct] = useState(DEFAULT_PCT)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    setNotAvailable(null)
    client
      .get('/api/backtest/curve', { params: { min_pct: 1, max_pct: 100, step_pct: 1 } })
      .then((response) => {
        if (cancelled) return
        if (response.data.available === false) {
          setNotAvailable(response.data.reason)
          return
        }
        setCurve(response.data)
      })
      .catch((err) => {
        if (!cancelled) setError(err.response?.data?.detail || err.message)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const chartData = useMemo(() => {
    if (!curve) return null
    return curve.map((point) => ({
      pct: Math.round(point.top_pct * 100),
      revenue_weighted: point.revenue_weighted.caught_pct,
      probability_only: point.probability_only.caught_pct,
      random: point.random.caught_pct,
    }))
  }, [curve])

  const current = curve ? curve[pct - 1] : null
  const vsRandom = current ? current.revenue_weighted.caught_pct - current.random.caught_pct : 0
  const vsProbOnly = current ? current.revenue_weighted.caught_pct - current.probability_only.caught_pct : 0

  return (
    <Section title="Backtest: Revenue Captured by Strategy" loading={loading} error={error} notAvailable={notAvailable}>
      {curve && current && chartData && (
        <div className="roi-calculator">
          <p className="muted small">
            Backtested against real, historical churn outcomes on the held-out test set — not a forecast for future
            customers. Drag the slider to see exactly how much churned revenue each strategy would have caught at
            that level of outreach; every point is computed directly, not interpolated.
          </p>

          <label className="roi-slider-label" htmlFor="roi-pct-slider">
            % of customers treated: <strong>{pct}%</strong> ({current.top_n.toLocaleString()} customers)
            <input
              id="roi-pct-slider"
              type="range"
              min={1}
              max={100}
              step={1}
              value={pct}
              onChange={(event) => setPct(Number(event.target.value))}
            />
          </label>

          <ResponsiveContainer width="100%" height={360}>
            <LineChart data={chartData} margin={{ top: 10, right: 16, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID_STROKE} />
              <XAxis
                dataKey="pct"
                type="number"
                domain={[1, 100]}
                label={{ value: '% of Customers Treated', position: 'bottom', offset: 0, style: CHART_AXIS_TICK }}
                tick={CHART_AXIS_TICK}
              />
              <YAxis
                domain={[0, 100]}
                label={{
                  value: '% of Churned Revenue Captured',
                  angle: -90,
                  position: 'insideLeft',
                  style: CHART_AXIS_TICK,
                }}
                tick={CHART_AXIS_TICK}
              />
              <Tooltip
                formatter={(value, name) => [`${value.toFixed(1)}%`, STRATEGY_LABELS[name]]}
                labelFormatter={(label) => `${label}% treated`}
                {...CHART_TOOLTIP_PROPS}
              />
              <Legend
                formatter={(value) => STRATEGY_LABELS[value]}
                wrapperStyle={{ paddingTop: 24, ...CHART_LEGEND_WRAPPER_STYLE }}
              />
              <ReferenceLine x={pct} stroke="var(--text-muted)" strokeDasharray="4 4" />
              {VALIDATED_CHECKPOINTS.map((checkpoint) => (
                <ReferenceDot
                  key={checkpoint}
                  x={checkpoint}
                  y={chartData[checkpoint - 1].revenue_weighted}
                  r={4}
                  fill={STRATEGY_COLORS.revenue_weighted}
                  stroke="var(--surface)"
                  strokeWidth={2}
                  isFront
                />
              ))}
              <Line
                type="monotone"
                dataKey="revenue_weighted"
                name="revenue_weighted"
                stroke={STRATEGY_COLORS.revenue_weighted}
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="probability_only"
                name="probability_only"
                stroke={STRATEGY_COLORS.probability_only}
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="random"
                name="random"
                stroke={STRATEGY_COLORS.random}
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>

          <div className="kpi-strip roi-kpi-strip">
            <div className="kpi-card kpi-card-primary">
              <span className="kpi-label">Revenue Captured (Revenue-Weighted)</span>
              <span className="kpi-value">{formatCurrency(current.revenue_weighted.caught_revenue)}</span>
              <span className="muted small">{current.revenue_weighted.caught_pct.toFixed(1)}% of churned revenue</span>
            </div>
            <div className="kpi-card">
              <span className="kpi-label">vs Random</span>
              <span className={`kpi-value ${liftClass(vsRandom)}`}>{formatLift(vsRandom)}</span>
            </div>
            <div className="kpi-card">
              <span className="kpi-label">vs Probability-Only</span>
              <span className={`kpi-value ${liftClass(vsProbOnly)}`}>{formatLift(vsProbOnly)}</span>
            </div>
            <div className="kpi-card">
              <span className="kpi-label">Total Churned Revenue (test set)</span>
              <span className="kpi-value">{formatCurrency(current.total_churned_revenue)}</span>
            </div>
          </div>

          {vsProbOnly < -0.5 && (
            <p className="caveat">
              At {pct}% treated, Revenue-Weighted is trailing Probability-Only by {Math.abs(vsProbOnly).toFixed(1)}{' '}
              points. The core project claim (Revenue-Weighted beats Probability-Only) was validated at 10%, 20%,
              and 30% treated — it does not hold at every single percentage, and this is one of the ranges where it
              doesn't.
            </p>
          )}

          <p className="muted small">
            Filled dots mark the 10%/20%/30% thresholds independently validated in <code>tests/test_backtest.py</code>{' '}
            — every other point on the slider is the same real backtest computation, not an interpolation between
            them.
          </p>
        </div>
      )}
    </Section>
  )
}
