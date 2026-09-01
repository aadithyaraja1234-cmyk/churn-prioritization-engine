import { useApiData } from '../hooks/useApiData'
import Section from './Section'

export default function KpiStrip() {
  const { data, loading, error } = useApiData('/api/model/metrics')
  const notAvailable = data?.available === false ? data.reason : null

  return (
    <Section title="Model Performance" loading={loading} error={error} notAvailable={notAvailable}>
      {data && (
        <div className="kpi-strip">
          <div className="kpi-card">
            <span className="kpi-label">ROC-AUC</span>
            <span className="kpi-value">{data.roc_auc?.toFixed(4)}</span>
          </div>
          <div className="kpi-card">
            <span className="kpi-label">PR-AUC</span>
            <span className="kpi-value">{data.pr_auc?.toFixed(4)}</span>
          </div>
          <div className="kpi-card">
            <span className="kpi-label">Customers (train / test)</span>
            <span className="kpi-value">
              {data.n_train} / {data.n_test}
            </span>
          </div>
          <div className="kpi-card">
            <span className="kpi-label">Churn Rate (test)</span>
            <span className="kpi-value">{(data.churn_rate_test * 100).toFixed(1)}%</span>
          </div>
        </div>
      )}
    </Section>
  )
}
