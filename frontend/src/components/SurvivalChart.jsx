import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useApiData } from '../hooks/useApiData'
import { CHART_AXIS_TICK, CHART_GRID_STROKE, CHART_TOOLTIP_PROPS } from '../styles/chartTheme'
import Section from './Section'

export default function SurvivalChart() {
  const { data, loading, error } = useApiData('/api/survival/segments')
  const notAvailable = data?.available === false ? data.reason : null

  const chartData =
    data && !notAvailable
      ? Object.entries(data).map(([contract, months]) => ({
          contract,
          months: months ?? 0,
          estimable: months !== null,
        }))
      : null

  return (
    <Section title="Median Survival Time by Contract" loading={loading} error={error} notAvailable={notAvailable}>
      {chartData && (
        <>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID_STROKE} />
              <XAxis dataKey="contract" tick={CHART_AXIS_TICK} />
              <YAxis
                label={{ value: 'Median tenure (months)', angle: -90, position: 'insideLeft', style: CHART_AXIS_TICK }}
                tick={CHART_AXIS_TICK}
              />
              <Tooltip
                formatter={(value, _name, props) =>
                  props.payload.estimable ? [`${value} months`, 'Median survival'] : ['Not yet estimable', 'Median survival']
                }
                {...CHART_TOOLTIP_PROPS}
              />
              <Bar dataKey="months" isAnimationActive={false}>
                {chartData.map((entry) => (
                  // was #2563EB/#D1D5DB - the "not yet estimable" gray was
                  // actually LIGHTER than the real bars (12.19:1 vs 3.48:1
                  // contrast), reading as MORE prominent than the real data
                  // on a dark background - inverted from its intended
                  // de-emphasized meaning. text-faint is properly muted.
                  <Cell
                    key={entry.contract}
                    fill={entry.estimable ? 'var(--color-accent)' : 'var(--color-text-faint)'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          {chartData
            .filter((entry) => !entry.estimable)
            .map((entry) => (
              <p key={entry.contract} className="muted small">
                {entry.contract}: Not yet estimable (churn too rare in this segment)
              </p>
            ))}
        </>
      )}
    </Section>
  )
}
