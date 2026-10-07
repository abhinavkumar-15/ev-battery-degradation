import { useState } from 'react'
import { Info } from 'lucide-react'
import { Line, LineChart, ResponsiveContainer, YAxis } from 'recharts'

/** Health colour bands relative to the end-of-life threshold. */
export const healthColor = (soh, eol = 80) => (soh >= eol + 10 ? '#10b981' : soh > eol ? '#f59e0b' : '#ef4444')

export function Gauge({ value, eol = 80, size = 150, label = 'SOH' }) {
  const r = size / 2 - 12, c = 2 * Math.PI * r
  const pct = Math.max(0, Math.min(100, value ?? 0)) / 100
  const col = healthColor(value, eol)
  // EOL tick on the ring
  const a = 2 * Math.PI * (eol / 100) - Math.PI / 2
  const tx = size / 2 + Math.cos(a) * r, ty = size / 2 + Math.sin(a) * r
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label={`${label} ${value?.toFixed?.(1)} percent`}>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--grid)" strokeWidth="10" />
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={col} strokeWidth="10" strokeLinecap="round"
        strokeDasharray={`${c * pct} ${c}`} transform={`rotate(-90 ${size / 2} ${size / 2})`} style={{ transition: 'stroke-dasharray .6s ease' }} />
      <circle cx={tx} cy={ty} r="3.5" fill="#ef4444" stroke="var(--tip-bg)" strokeWidth="1.5"><title>End-of-life threshold {eol}%</title></circle>
      <text x="50%" y="48%" textAnchor="middle" className="num" style={{ fontSize: size * 0.2, fontWeight: 650, fill: 'var(--tip-fg)' }}>{value == null ? '—' : value.toFixed(1)}%</text>
      <text x="50%" y="64%" textAnchor="middle" style={{ fontSize: 11, fill: 'var(--axis)' }}>{label}</text>
    </svg>
  )
}

export const Sparkline = ({ data, dataKey, color = '#14b8a6', height = 36 }) => (
  <div style={{ height }} className="w-full" aria-hidden>
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data}><YAxis hide domain={['auto', 'auto']} /><Line dataKey={dataKey} stroke={color} strokeWidth={1.6} dot={false} isAnimationActive={false} /></LineChart>
    </ResponsiveContainer>
  </div>
)

export function InfoTip({ text }) {
  const [open, setOpen] = useState(false)
  return (
    <span className="relative inline-flex align-middle">
      <button type="button" aria-label="More information" onClick={() => setOpen(!open)} onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}
        className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"><Info className="h-3.5 w-3.5" /></button>
      {open && <span role="tooltip" className="absolute left-5 top-0 z-30 w-64 rounded-lg border border-slate-200 bg-white p-2.5 text-[11px] font-normal leading-snug text-slate-600 shadow-lg dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300">{text}</span>}
    </span>
  )
}

export const Segmented = ({ value, onChange, options }) => (
  <div role="tablist" className="inline-flex rounded-lg border border-slate-200 bg-slate-100 p-0.5 dark:border-slate-700 dark:bg-slate-800">
    {options.map((o) => (
      <button key={o.value} role="tab" aria-selected={value === o.value} onClick={() => onChange(o.value)}
        className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${value === o.value ? 'bg-white text-slate-900 shadow-sm dark:bg-slate-950 dark:text-white' : 'text-slate-500 hover:text-slate-800 dark:hover:text-slate-200'}`}>{o.label}</button>
    ))}
  </div>
)

export const Bar2 = ({ value, max = 100, color = '#14b8a6' }) => (
  <div className="h-1.5 w-full rounded-full bg-slate-200 dark:bg-slate-800"><div className="h-1.5 rounded-full" style={{ width: `${Math.max(0, Math.min(100, (value / max) * 100))}%`, background: color }} /></div>
)

export function downloadCsv(filename, rows, cols) {
  const esc = (v) => (v == null ? '' : String(v).includes(',') ? `"${v}"` : v)
  const csv = [cols.map((c) => c.label).join(','), ...rows.map((r) => cols.map((c) => esc(r[c.key])).join(','))].join('\n')
  const a = document.createElement('a')
  a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' })); a.download = filename; a.click(); URL.revokeObjectURL(a.href)
}

export const statusTone = (s) => (s?.startsWith('End') ? 'bad' : s === 'Healthy' ? 'good' : 'warn')
