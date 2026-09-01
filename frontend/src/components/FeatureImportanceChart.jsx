import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useApiData } from '../hooks/useApiData'
import { CHART_AXIS_TICK, CHART_GRID_STROKE, CHART_TOOLTIP_PROPS } from '../styles/chartTheme'
import Section from './Section'

export default function FeatureImportanceChart() {
  const { data, loading, error } = useApiData('/api/model/importance')
  const notAvailable = data?.available === false ? data.reason : null
  const chartData = Array.isArray(data) ? [...data].sort((a, b) => a.importance - b.importance) : null

  return (
    <Section title="Feature Importance" loading={loading} error={error} notAvailable={notAvailable}>
      {chartData && (
        <ResponsiveContainer width="100%" height={Math.max(300, chartData.length * 28)}>
          <BarChart data={chartData} layout="vertical" margin={{ left: 100 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID_STROKE} />
            <XAxis type="number" tick={CHART_AXIS_TICK} />
            <YAxis type="category" dataKey="feature" width={140} tick={CHART_AXIS_TICK} />
            <Tooltip {...CHART_TOOLTIP_PROPS} />
            <Bar dataKey="importance" fill="var(--color-accent)" isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      )}
    </Section>
  )
}
