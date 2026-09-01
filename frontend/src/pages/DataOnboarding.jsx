import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import AppHeader from '../components/AppHeader'
import TrainingStatusIndicator from '../components/TrainingStatusIndicator'
import client from '../api/client'
import { useToast } from '../hooks/useToast'

const WIZARD_STEPS = [
  { key: 'upload', label: '1. Upload' },
  { key: 'mapping', label: '2. Map Columns' },
  { key: 'report', label: '3. Review Results' },
]

function friendlyError(err) {
  const detail = err.response?.data?.detail
  if (typeof detail === 'string') return detail
  return err.message || 'Something went wrong - please try again.'
}

function verdictFor(report) {
  if (!report) return null
  if (!report.can_proceed) return 'blocked'
  if (report.data_sufficiency.warning) return 'warning'
  return 'ready'
}

const VERDICT_COPY = {
  ready: { icon: '✓', label: 'Ready to go', className: 'verdict-banner-ready' },
  warning: { icon: '!', label: 'Needs attention', className: 'verdict-banner-warning' },
  blocked: { icon: '✗', label: "Can't proceed yet", className: 'verdict-banner-blocked' },
}

export default function DataOnboarding() {
  const { showToast } = useToast()
  const fileInputRef = useRef(null)

  const [wizardStep, setWizardStep] = useState('upload')
  const [resumed, setResumed] = useState(false)
  const [alreadyReady, setAlreadyReady] = useState(false)

  const [dragActive, setDragActive] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState(null)

  const [upload, setUpload] = useState(null)
  const [mapping, setMapping] = useState({})

  const [validating, setValidating] = useState(false)
  const [validateError, setValidateError] = useState(null)
  const [report, setReport] = useState(null)
  const [acknowledged, setAcknowledged] = useState(false)
  const [confirming, setConfirming] = useState(false)
  const [confirmed, setConfirmed] = useState(false)

  const [trainingJobId, setTrainingJobId] = useState(null)
  const [startingTraining, setStartingTraining] = useState(false)

  useEffect(() => {
    let cancelled = false
    client
      .get('/api/companies/onboarding-status')
      .then((response) => {
        if (cancelled) return
        const status = response.data
        if (status.step === 'ready') {
          setAlreadyReady(true)
        }
        if ((status.step === 'data_uploaded' || status.step === 'data_validated') && status.last_upload_id) {
          return client.get(`/api/onboarding/upload/${status.last_upload_id}`).then((uploadResponse) => {
            if (cancelled) return
            setUpload(uploadResponse.data)
            const initialMapping = {}
            for (const column of uploadResponse.data.columns) {
              initialMapping[column] = uploadResponse.data.suggested_mapping[column]?.suggested_role || 'ignore'
            }
            setMapping(initialMapping)
            setWizardStep('mapping')
            setResumed(true)
          })
        }
      })
      .catch(() => {}) // no status yet (e.g. a pre-existing demo tenant) - just start fresh
    return () => {
      cancelled = true
    }
  }, [])

  async function handleFile(file) {
    if (!file) return
    setUploading(true)
    setUploadError(null)
    try {
      const formData = new FormData()
      formData.append('file', file)
      const response = await client.post('/api/onboarding/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setUpload(response.data)
      const initialMapping = {}
      for (const column of response.data.columns) {
        initialMapping[column] = response.data.suggested_mapping[column]?.suggested_role || 'ignore'
      }
      setMapping(initialMapping)
      setWizardStep('mapping')
    } catch (err) {
      setUploadError(friendlyError(err))
    } finally {
      setUploading(false)
    }
  }

  function handleDrop(event) {
    event.preventDefault()
    setDragActive(false)
    const file = event.dataTransfer.files?.[0]
    if (file) handleFile(file)
  }

  async function handleValidate() {
    setValidating(true)
    setValidateError(null)
    setAcknowledged(false)
    setConfirmed(false)
    try {
      const response = await client.post('/api/onboarding/validate', {
        upload_id: upload.upload_id,
        column_mapping: mapping,
      })
      setReport(response.data)
      setWizardStep('report')
    } catch (err) {
      setValidateError(friendlyError(err))
    } finally {
      setValidating(false)
    }
  }

  function handleStartOver() {
    setWizardStep('upload')
    setUpload(null)
    setMapping({})
    setReport(null)
    setUploadError(null)
    setValidateError(null)
    setAcknowledged(false)
    setConfirmed(false)
    setResumed(false)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  function handleBackToUpload() {
    handleStartOver()
  }

  function handleBackToMapping() {
    setReport(null)
    setWizardStep('mapping')
  }

  async function handleConfirm() {
    setConfirming(true)
    try {
      await client.post('/api/onboarding/confirm')
      setConfirmed(true)
      showToast('Onboarding marked as ready')
    } catch {
      showToast('Could not save - please try again')
    } finally {
      setConfirming(false)
    }
  }

  async function handleStartTraining() {
    setStartingTraining(true)
    try {
      const response = await client.post('/api/training/start', { upload_id: upload.upload_id })
      setTrainingJobId(response.data.job_id)
      showToast('Training job started')
    } catch {
      showToast('Could not start training - please try again')
    } finally {
      setStartingTraining(false)
    }
  }

  const verdict = verdictFor(report)
  const verdictCopy = verdict ? VERDICT_COPY[verdict] : null
  const currentStepIndex = WIZARD_STEPS.findIndex((s) => s.key === wizardStep)

  return (
    <div className="dashboard">
      <AppHeader title="Add Data" />

      {alreadyReady && (
        <div className="onboarding-resume-banner">
          <span>✓ Onboarding already complete for your company - you can upload additional data anytime.</span>
        </div>
      )}
      {resumed && !alreadyReady && (
        <div className="onboarding-resume-banner">
          <span>Resumed your in-progress upload ({upload?.filename}) - pick up where you left off.</span>
        </div>
      )}

      <nav className="onboarding-progress">
        {WIZARD_STEPS.map((step, index) => (
          <div
            key={step.key}
            className={`onboarding-progress-step ${
              index < currentStepIndex
                ? 'onboarding-progress-done'
                : index === currentStepIndex
                  ? 'onboarding-progress-active'
                  : ''
            }`}
          >
            <div className="onboarding-progress-bar" />
            <span className="onboarding-progress-label">{step.label}</span>
          </div>
        ))}
      </nav>

      {wizardStep === 'upload' && (
        <section className="dashboard-section">
          <h2>Upload a CSV</h2>
          <p className="muted">
            Upload a customer-level CSV to preview how it maps to this system's known fields and
            whether it has enough signal for reliable churn modeling. This step does not train a
            model - it's a preview only.
          </p>
          <div
            className={`upload-dropzone${dragActive ? ' upload-dropzone-active' : ''}`}
            onDragOver={(event) => {
              event.preventDefault()
              setDragActive(true)
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <p>{uploading ? 'Uploading…' : 'Drag a .csv file here, or click to choose one'}</p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,text/csv"
              hidden
              onChange={(event) => handleFile(event.target.files?.[0])}
            />
          </div>
          {uploadError && <p className="error-text upload-error">{uploadError}</p>}
        </section>
      )}

      {wizardStep === 'mapping' && upload && (
        <section className="dashboard-section">
          <h2>Map Your Columns</h2>
          <p className="muted">
            {upload.filename} — {upload.row_count.toLocaleString()} rows, {upload.columns.length} columns.
            We've suggested a role for each column by name similarity; confirm or change every one below.
          </p>
          <table className="mapping-table">
            <thead>
              <tr>
                <th>Your Column</th>
                <th>Sample Value</th>
                <th>Role</th>
              </tr>
            </thead>
            <tbody>
              {upload.columns.map((column) => (
                <tr key={column}>
                  <td>{column}</td>
                  <td className="mapping-sample">{String(upload.preview_rows[0]?.[column] ?? '')}</td>
                  <td>
                    <select
                      value={mapping[column] || 'ignore'}
                      onChange={(event) => setMapping({ ...mapping, [column]: event.target.value })}
                    >
                      {upload.valid_roles.map((role) => (
                        <option key={role} value={role}>
                          {role.replaceAll('_', ' ')}
                        </option>
                      ))}
                    </select>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {validateError && <p className="error-text">{validateError}</p>}
          <div className="onboarding-actions">
            <button type="button" onClick={handleBackToUpload} disabled={validating}>
              Back
            </button>
            <button type="button" onClick={handleValidate} disabled={validating}>
              {validating ? 'Validating…' : 'Validate'}
            </button>
          </div>
        </section>
      )}

      {wizardStep === 'report' && report && (
        <section className="dashboard-section">
          <h2>Validation Report</h2>

          <div className={`verdict-banner ${verdictCopy.className}`}>
            <span className="verdict-banner-icon">{verdictCopy.icon}</span>
            <span>{verdictCopy.label}</span>
          </div>

          <div className="validation-checks">
            {report.checks.map((check) => (
              <div key={check.name} className={`validation-check ${check.passed ? 'validation-check-pass' : 'validation-check-fail'}`}>
                <span className="validation-check-icon">{check.passed ? '✓' : '✗'}</span>
                <span>{check.detail}</span>
              </div>
            ))}
          </div>

          <div className={`data-sufficiency-card ${report.data_sufficiency.warning ? 'data-sufficiency-warning' : 'data-sufficiency-ok'}`}>
            <h3>{report.data_sufficiency.warning ? 'Data Sufficiency: Thin Data Warning' : 'Data Sufficiency: Sufficient'}</h3>
            <dl className="data-sufficiency-stats">
              <div>
                <dt>Rows</dt>
                <dd>
                  {report.data_sufficiency.row_count.toLocaleString()} / recommended {report.data_sufficiency.min_recommended_row_count.toLocaleString()}
                </dd>
              </div>
              <div>
                <dt>Usable Predictive Columns</dt>
                <dd>
                  {report.data_sufficiency.feature_richness_score} / recommended {report.data_sufficiency.min_recommended_features}
                </dd>
              </div>
            </dl>
            {report.data_sufficiency.guidance && <p className="data-sufficiency-guidance">{report.data_sufficiency.guidance}</p>}
          </div>

          {confirmed ? (
            <div className="onboarding-actions">
              <p className="muted">Saved. Your workspace is marked as ready.</p>
              {trainingJobId ? (
                <TrainingStatusIndicator jobId={trainingJobId} />
              ) : (
                <button type="button" onClick={handleStartTraining} disabled={startingTraining}>
                  {startingTraining ? 'Starting…' : 'Start Training'}
                </button>
              )}
              <Link to="/" className="register-welcome-button register-welcome-button-primary">
                Go to Dashboard
              </Link>
            </div>
          ) : (
            <div className="onboarding-actions">
              <button type="button" onClick={handleBackToMapping}>
                Back
              </button>
              <button type="button" onClick={handleStartOver}>
                Start Over
              </button>
              {report.can_proceed && report.data_sufficiency.requires_acknowledgment && (
                <>
                  <label className="onboarding-acknowledge">
                    <input type="checkbox" checked={acknowledged} onChange={(event) => setAcknowledged(event.target.checked)} />
                    I understand results may be less reliable with this data and want to proceed anyway.
                  </label>
                  <button type="button" disabled={!acknowledged || confirming} onClick={handleConfirm}>
                    {confirming ? 'Saving…' : 'Proceed Anyway'}
                  </button>
                </>
              )}
              {report.can_proceed && !report.data_sufficiency.requires_acknowledgment && (
                <button type="button" disabled={confirming} onClick={handleConfirm}>
                  {confirming ? 'Saving…' : 'Continue'}
                </button>
              )}
              {!report.can_proceed && (
                <p className="error-text">Fix the failed checks above and re-upload before this data can be used.</p>
              )}
            </div>
          )}
        </section>
      )}
    </div>
  )
}
