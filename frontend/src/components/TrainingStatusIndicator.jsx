import { useEffect, useRef, useState } from 'react'
import client from '../api/client'

const POLL_INTERVAL_MS = 2000

const STATUS_COPY = {
  queued: 'Training in progress… (queued)',
  running: 'Training in progress… (running)',
  succeeded: 'Training complete',
  failed: 'Training failed',
}

// Order/labels for the per-module breakdown below - classifier first (it's
// not itself an entry in result_metadata_json.modules, its numbers live at
// the top level), then every module src/models/tenant_training.py's
// _run_optional_modules() runs, in that same order: the original four
// Stage 2b modules, then priority_ranking/backtest/customer_timeline/
// scenario_simulator (Stage 2c - see that function's own docstring).
const MODULE_LABELS = {
  survival: 'Survival',
  segments: 'Segments',
  anomalies: 'Anomalies',
  clv: 'CLV',
  priority_ranking: 'Priority Ranking',
  backtest: 'Backtest',
  customer_timeline: 'Customer Timeline',
  scenario_simulator: 'Scenario Simulator',
}

function classifierSummary(metadata) {
  const auc = `AUC ${metadata.roc_auc?.toFixed(3)}`
  if (metadata.is_sane) return { passed: true, detail: `done (${auc})` }
  return { passed: false, detail: `below confidence threshold, not enabled (${auc})` }
}

// One real, honest line per module - "done (metric)" for a pass, a named
// reason for a fail/skip, never just "pending" once the job has actually
// finished (see result_metadata_json.modules' shape, api/training.py).
function moduleSummary(name, result) {
  if (!result.ran) {
    return { passed: false, skipped: true, detail: `skipped — ${result.reason}` }
  }
  if (!result.passed) {
    return { passed: false, detail: `below confidence threshold, not enabled — ${result.reason}` }
  }
  const metrics = result.metrics || {}
  switch (name) {
    case 'survival':
      return { passed: true, detail: `done (c-index ${metrics.c_index?.toFixed(3)})` }
    case 'segments':
      return { passed: true, detail: `done (k=${metrics.chosen_k}, silhouette ${metrics.silhouette?.toFixed(3)})` }
    case 'anomalies':
      return { passed: true, detail: `done (${(metrics.flagged_fraction * 100).toFixed(1)}% flagged)` }
    case 'clv':
      return { passed: true, detail: `done (R² ${metrics.r2?.toFixed(3)})` }
    case 'priority_ranking':
      return {
        passed: true,
        detail: `done (revenue vs. probability correlation ${metrics.revenue_vs_probability_spearman_correlation?.toFixed(3)})`,
      }
    case 'backtest':
      return { passed: true, detail: `done (${metrics.n_churned_test} churned test customers)` }
    case 'customer_timeline':
      return { passed: true, detail: 'done' }
    case 'scenario_simulator':
      return { passed: true, detail: `done (${(metrics.probe_affected_fraction * 100).toFixed(1)}% affected by probe scenario)` }
    default:
      return { passed: true, detail: 'done' }
  }
}

// Polls GET /api/training/status/{jobId} every few seconds until the job
// reaches a terminal status (succeeded/failed), then stops. This
// component only reflects whatever status the backend actually reports,
// never guesses or fast-forwards past what the API has confirmed - the
// per-module breakdown below only appears once result_metadata_json is
// actually populated (the backend job is one atomic unit today: all five
// modules run sequentially inside a single background thread before the
// job is marked "succeeded", so there is no true module-by-module status
// to poll mid-run - this renders the real, complete breakdown the moment
// it lands, not a live progress bar).
export default function TrainingStatusIndicator({ jobId }) {
  const [job, setJob] = useState(null)
  const timerRef = useRef(null)

  useEffect(() => {
    if (!jobId) return
    let cancelled = false

    function poll() {
      client
        .get(`/api/training/status/${jobId}`)
        .then((response) => {
          if (cancelled) return
          setJob(response.data)
          if (response.data.status === 'succeeded' || response.data.status === 'failed') return
          timerRef.current = setTimeout(poll, POLL_INTERVAL_MS)
        })
        .catch(() => {
          if (!cancelled) timerRef.current = setTimeout(poll, POLL_INTERVAL_MS)
        })
    }
    poll()

    return () => {
      cancelled = true
      clearTimeout(timerRef.current)
    }
  }, [jobId])

  const status = job?.status || 'queued'
  const isActive = status === 'queued' || status === 'running'
  const metadata = job?.result_metadata_json

  return (
    <div
      className={`training-status ${
        status === 'succeeded' ? 'training-status-succeeded' : status === 'failed' ? 'training-status-failed' : 'training-status-active'
      }`}
    >
      <div className="training-status-row">
        {isActive && <span className="training-status-spinner" aria-hidden="true" />}
        <span>{STATUS_COPY[status] || status}</span>
      </div>
      {status === 'failed' && job?.error_message && <p className="training-status-error">{job.error_message}</p>}
      {status === 'succeeded' && metadata && (
        <div className="training-status-modules">
          {[
            ['classifier', classifierSummary(metadata)],
            ...Object.keys(MODULE_LABELS)
              .filter((name) => metadata.modules?.[name])
              .map((name) => [name, moduleSummary(name, metadata.modules[name])]),
          ].map(([name, summary]) => (
            <div
              key={name}
              className={`validation-check ${
                summary.passed ? 'validation-check-pass' : summary.skipped ? 'training-status-module-skipped' : 'validation-check-fail'
              }`}
            >
              <span className="validation-check-icon">{summary.passed ? '✓' : summary.skipped ? '—' : '✗'}</span>
              <span>
                <strong>{name === 'classifier' ? 'Classifier' : MODULE_LABELS[name]}:</strong> {summary.detail}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
