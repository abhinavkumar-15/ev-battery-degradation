// Thin API client. Every number shown in the UI comes from these endpoints.
const rawBase = import.meta.env.VITE_API_BASE ?? ''
const BASE = rawBase.replace(/\/+$/, '')

async function request(path, opts) {
  let res
  try {
    res = await fetch(BASE + path, { headers: { 'Content-Type': 'application/json' }, ...opts })
  } catch {
    throw new Error('Cannot reach the VoltGuard API. Is the backend running on port 8000?')
  }
  if (!res.ok) {
    let msg = `Request failed (${res.status})`
    try {
      const j = await res.json()
      const d = j.detail
      msg = typeof d === 'string' ? d : d?.error ? `${d.error}: ${(d.details || []).join('; ')}` : msg
    } catch { /* non-JSON error body */ }
    const e = new Error(msg); e.status = res.status; throw e
  }
  return res.json()
}

const qs = (o) => {
  const p = new URLSearchParams()
  Object.entries(o).forEach(([k, v]) => v !== undefined && v !== null && v !== '' && p.set(k, v))
  const s = p.toString(); return s ? `?${s}` : ''
}
const post = (path, body) => request(path, { method: 'POST', body: JSON.stringify(body) })

export const api = {
  health: () => request('/api/health'),
  datasets: () => request('/api/datasets'),
  batteries: (ds) => request(`/api/datasets/${ds}/batteries`),
  scenarioLimits: (id, dataset, cycle) => request(`/api/scenario/limits${qs({ battery_id: id, dataset, cycle })}`),
  fleet: (ds) => request(`/api/datasets/${ds}/fleet`),
  horizonCurve: (dataset) => request(`/api/models/horizons${qs({ dataset })}`),
  datasetSummary: (ds) => request(`/api/datasets/${ds}/summary`),
  summary: (id, dataset) => request(`/api/batteries/${id}/summary${qs({ dataset })}`),
  history: (id, dataset, o = {}) => request(`/api/batteries/${id}/history${qs({ dataset, ...o })}`),
  predictSoh: (b) => post('/api/predict/soh', b),
  predictDeg: (b) => post('/api/predict/degradation', b),
  predictRul: (b) => post('/api/predict/rul', b),
  models: (dataset) => request(`/api/models${qs({ dataset })}`),
  performance: (dataset, horizon) => request(`/api/models/performance${qs({ dataset, horizon })}`),
  explain: (dataset) => request(`/api/explainability${qs({ dataset })}`),
  explainBattery: (id, dataset, cycle) => request(`/api/explainability/${id}${qs({ dataset, cycle })}`),
  evPresets: () => request('/api/ev-calculator/presets'),
  evPredict: (b) => post('/api/ev-calculator/predict', b),
}

