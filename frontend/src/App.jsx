import { useState, useEffect, useCallback } from 'react'
import { fetchHealth, fetchDrift, simulateTransactions, triggerLocalRetrain } from './api'
import './App.css'

function DriftGauge({ score, threshold }) {
  const pct = Math.min(100, (score / threshold) * 100)
  const color = score >= threshold ? 'var(--danger)' : score >= threshold * 0.7 ? 'var(--warning)' : 'var(--success)'

  return (
    <div className="gauge">
      <div className="gauge-track">
        <div className="gauge-fill" style={{ width: `${pct}%`, background: color }} />
        <div className="gauge-marker" style={{ left: '100%' }} title={`Threshold: ${threshold}`} />
      </div>
      <div className="gauge-labels">
        <span>0</span>
        <span className="mono">{(score * 100).toFixed(1)}% drifted columns</span>
        <span className="mono">{(threshold * 100).toFixed(0)}% threshold</span>
      </div>
    </div>
  )
}

function TransactionRow({ txn, index }) {
  return (
    <div className={`txn-row ${txn.is_fraud ? 'fraud' : 'legit'}`}>
      <span className="mono txn-id">#{String(index + 1).padStart(3, '0')}</span>
      <span className="mono txn-prob">{(txn.fraud_probability * 100).toFixed(1)}%</span>
      <span className={`badge ${txn.is_fraud ? 'badge-danger' : 'badge-ok'}`}>
        {txn.is_fraud ? 'FRAUD' : 'OK'}
      </span>
      <span className="mono txn-ver muted">{txn.model_version}</span>
    </div>
  )
}

export default function App() {
  const [health, setHealth] = useState(null)
  const [drift, setDrift] = useState(null)
  const [transactions, setTransactions] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [batchSize, setBatchSize] = useState(25)

  const refresh = useCallback(async () => {
    try {
      const [h, d] = await Promise.all([fetchHealth(), fetchDrift()])
      setHealth(h)
      setDrift(d)
      setError(null)
    } catch (e) {
      setError(e.message)
    }
  }, [])

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, 5000)
    return () => clearInterval(id)
  }, [refresh])

  async function handleSimulate(injectDrift) {
    setLoading(true)
    setError(null)
    try {
      const results = await simulateTransactions(batchSize, injectDrift)
      setTransactions((prev) => [...results, ...prev].slice(0, 50))
      await refresh()
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function handleRetrain() {
    setLoading(true)
    try {
      await triggerLocalRetrain('change-me-in-production')
      await refresh()
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div>
          <p className="eyebrow">MLOps · Model Observability</p>
          <h1>Autonomous ML Platform</h1>
          <p className="subtitle">Real-Time Fraud Detection with Automated Drift Retraining</p>
        </div>
        <div className="header-status">
          {health && (
            <span className={`badge ${health.model_loaded ? 'badge-ok' : 'badge-warn'}`}>
              Model {health.model_loaded ? 'Live' : 'Loading'}
            </span>
          )}
          {drift?.drift_detected && <span className="badge badge-danger">Drift Alert</span>}
          {drift?.retrain_triggered && <span className="badge badge-warn">Retrain Queued</span>}
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <div className="grid">
        <section className="card drift-panel">
          <h2>Evidently Drift Monitor</h2>
          <p className="muted">Compares live transaction stream against reference distribution</p>
          {drift && (
            <>
              <DriftGauge score={drift.drift_score} threshold={drift.threshold} />
              <div className="stats-row">
                <div className="stat">
                  <span className="stat-label">Buffer</span>
                  <span className="stat-value mono">{drift.buffer_size} / {drift.window_size}</span>
                </div>
                <div className="stat">
                  <span className="stat-label">Last Check</span>
                  <span className="stat-value mono small">
                    {drift.last_check ? new Date(drift.last_check).toLocaleTimeString() : '—'}
                  </span>
                </div>
                <div className="stat">
                  <span className="stat-label">Model Version</span>
                  <span className="stat-value mono small">{health?.model_version ?? '—'}</span>
                </div>
              </div>
            </>
          )}
        </section>

        <section className="card controls-panel">
          <h2>Transaction Simulator</h2>
          <p className="muted">Stream synthetic credit-card transactions into the inference API</p>
          <div className="control-row">
            <label>
              Batch size
              <input
                type="number"
                min={1}
                max={200}
                value={batchSize}
                onChange={(e) => setBatchSize(Number(e.target.value))}
              />
            </label>
          </div>
          <div className="btn-row">
            <button className="btn-primary" disabled={loading} onClick={() => handleSimulate(false)}>
              Stream Normal Data
            </button>
            <button className="btn-danger" disabled={loading} onClick={() => handleSimulate(true)}>
              Inject Drifted Data
            </button>
          </div>
          <p className="hint">
            Use <strong>Inject Drifted Data</strong> to mildly shift amount and a subset of PCA features (~30–40% column drift) and trigger retraining.
          </p>
          {drift?.drift_detected && (
            <button className="btn-ghost retrain-btn" disabled={loading} onClick={handleRetrain}>
              Trigger Local Retrain (Webhook Fallback)
            </button>
          )}
        </section>
      </div>

      <section className="card feed-panel">
        <div className="feed-header">
          <h2>Live Predictions</h2>
          <span className="muted">{transactions.length} recent</span>
        </div>
        <div className="txn-header">
          <span>ID</span>
          <span>Fraud Prob</span>
          <span>Verdict</span>
          <span>Model</span>
        </div>
        <div className="txn-list">
          {transactions.length === 0 ? (
            <p className="empty muted">Simulate transactions to see real-time fraud scores</p>
          ) : (
            transactions.map((txn, i) => <TransactionRow key={txn.transaction_id} txn={txn} index={i} />)
          )}
        </div>
      </section>

      <footer className="footer muted">
        FastAPI · LightGBM · Evidently AI · MLflow · Docker · GitHub Actions
      </footer>
    </div>
  )
}
