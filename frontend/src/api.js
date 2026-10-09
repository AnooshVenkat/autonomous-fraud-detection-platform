const API_BASE = import.meta.env.VITE_API_URL || '/api'

async function parseError(res, fallback) {
  try {
    const body = await res.json()
    const detail = body.detail || body.message
    if (Array.isArray(detail)) {
      return detail.map((d) => d.msg || JSON.stringify(d)).join('; ') || fallback
    }
    return detail || fallback
  } catch {
    return fallback
  }
}

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`)
  if (!res.ok) throw new Error(await parseError(res, 'Health check failed'))
  return res.json()
}

export async function fetchDrift() {
  const res = await fetch(`${API_BASE}/drift`)
  if (!res.ok) throw new Error(await parseError(res, 'Drift status failed'))
  return res.json()
}

export async function simulateTransactions(count, injectDrift) {
  const res = await fetch(`${API_BASE}/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ count, inject_drift: injectDrift }),
  })
  if (!res.ok) throw new Error(await parseError(res, 'Simulation failed'))
  return res.json()
}

export async function triggerLocalRetrain(secret) {
  const res = await fetch(`${API_BASE}/retrain/webhook`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ secret }),
  })
  if (!res.ok) throw new Error(await parseError(res, 'Retrain failed'))
  return res.json()
}
