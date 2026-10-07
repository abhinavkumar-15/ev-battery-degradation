import { useState } from 'react'
import { Async, Badge, Card, DemoBanner, Empty, Field, NumberInput, PageTitle, fmt, useAsync } from '../components/ui.jsx'
import { Download } from 'lucide-react'
import { downloadCsv } from '../components/viz.jsx'
import { useApp } from '../context/AppContext.jsx'
import { api } from '../services/api.js'

const COLS = [['cycle', 'Cycle', 0], ['capacity_ah', 'Capacity (Ah)', 3], ['soh', 'SOH (%)', 2], ['degradation_rate', 'Rate (%/cyc)', 3], ['dis_t_mean', 'T mean (°C)', 2], ['dis_i_mean', 'I mean (A)', 2],
  ['dis_v_mean', 'V mean (V)', 3], ['dis_duration_s', 'Dis. time (s)', 0], ['re_ohm', 'Re (Ω)', 4], ['rct_ohm', 'Rct (Ω)', 4]]
const PAGE = 15

export default function DatasetExplorer() {
  const { dataset, dsId, battery, batteries, setBattery } = useApp()
  const [lo, setLo] = useState(1); const [hi, setHi] = useState(''); const [page, setPage] = useState(0)
  const sum = useAsync(() => api.datasetSummary(dsId), [dsId])
  const hist = useAsync(() => api.history(battery, dsId), [battery, dsId])
  const rows = (hist.data?.records || []).filter((r) => r.cycle >= (lo || 1) && (hi === '' || r.cycle <= hi))
  const pages = Math.max(1, Math.ceil(rows.length / PAGE))

  return (
    <div className="mx-auto max-w-6xl">
      <PageTitle title="Dataset Explorer" subtitle={`Source dataset: ${dsId} · every record carries dataset + battery id`} />
      <DemoBanner dataset={dataset} />
      <div className="grid gap-4 lg:grid-cols-3">
        <Card title="Batteries" subtitle="Click a row to select">
          <table className="w-full text-left text-xs"><thead className="text-slate-500"><tr><th className="py-1">ID</th><th>Cycles</th><th>Final SOH</th><th>Amb. °C</th></tr></thead>
            <tbody>{batteries.map((b) => (
              <tr key={b.id} onClick={() => { setBattery(b.id); setPage(0) }} className={`cursor-pointer border-t border-slate-100 dark:border-slate-800 ${b.id === battery ? 'bg-brand-50 dark:bg-brand-700/20' : ''}`}>
                <td className="py-1.5 font-medium">{b.id}</td><td className="num">{b.cycles}</td><td className="num">{fmt(b.last_soh, 1)}</td><td className="num">{fmt(b.ambient_temp_c, 0)}</td></tr>))}</tbody></table>
        </Card>
        <Card className="lg:col-span-2" title="Summary statistics" subtitle="All cycles, all batteries">
          <Async state={sum}>{(s) => (<div className="overflow-x-auto"><table className="w-full text-left text-xs"><thead className="text-slate-500"><tr><th className="py-1">Column</th><th>Count</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr></thead>
            <tbody>{s.stats.map((r) => <tr key={r.column} className="num border-t border-slate-100 dark:border-slate-800"><td className="py-1.5 font-mono">{r.column}</td><td>{r.count}</td><td>{fmt(r.mean, 3)}</td><td>{fmt(r.std, 3)}</td><td>{fmt(r.min, 3)}</td><td>{fmt(r.max, 3)}</td></tr>)}</tbody></table></div>)}</Async>
        </Card>
        <Card className="lg:col-span-3" title="Data quality" subtitle="What cleaning removed or flagged — nothing is dropped silently">
          <Async state={sum}>{(s) => {
            const q = s.quality; const drops = Object.entries(q.dropped), flags = Object.entries(q.flags), miss = Object.entries(q.missing_fraction)
            return (<div className="grid gap-4 text-xs sm:grid-cols-3">
              <div><div className="mb-1 font-medium">Rows</div><div className="num">{q.rows_in} in → {q.rows_out} used</div>
                <div className="mt-2 font-medium">Reference capacity</div><div>{q.reference.mode}{q.reference.rated_capacity_ah ? ` (${q.reference.rated_capacity_ah} Ah)` : ''}</div>
                <div className="mt-2 font-medium">Capacity source</div>{Object.entries(s.capacity_source).map(([k, v]) => <Badge key={k} tone={k === 'synthetic' ? 'warn' : 'neutral'}>{k}: {v}</Badge>)}</div>
              <div><div className="mb-1 font-medium">Dropped / flagged</div>{drops.length + flags.length === 0 ? <span className="text-emerald-600">nothing</span> :
                [...drops, ...flags].map(([k, v]) => <div key={k}>{k}: <b>{Array.isArray(v) ? v.join(', ') : v}</b></div>)}</div>
              <div><div className="mb-1 font-medium">Missing fraction</div>{miss.length === 0 ? <span className="text-emerald-600">no missing values</span> : miss.slice(0, 8).map(([k, v]) => <div key={k}><span className="font-mono">{k}</span>: {(v * 100).toFixed(1)}%</div>)}</div></div>)
          }}</Async>
        </Card>
        <Card className="lg:col-span-3" title={`Records — ${battery}`} subtitle="measured and derived columns"
          right={<div className="flex items-end gap-2"><button className="mb-0.5 inline-flex items-center gap-1 rounded-md border border-slate-300 px-2.5 py-1.5 text-xs font-medium hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800" onClick={() => downloadCsv(`${dsId}_${battery}_cycles.csv`, rows, COLS.map(([key, label]) => ({ key, label })))}><Download className="h-3.5 w-3.5" /> CSV</button><div className="w-20"><Field label="From"><NumberInput min={1} value={lo} onChange={(v) => { setLo(v); setPage(0) }} /></Field></div><div className="w-20"><Field label="To"><NumberInput min={1} value={hi} onChange={(v) => { setHi(v); setPage(0) }} /></Field></div></div>}>
          <Async state={hist}>{() => rows.length === 0 ? <Empty title="No records match" hint="Adjust the cycle filter." /> : (<>
            <div className="overflow-x-auto"><table className="w-full text-left text-xs"><thead className="text-slate-500"><tr>{COLS.map(([, h]) => <th key={h} className="whitespace-nowrap py-1 pr-3">{h}</th>)}</tr></thead>
              <tbody>{rows.slice(page * PAGE, page * PAGE + PAGE).map((r) => <tr key={r.cycle} className="num border-t border-slate-100 dark:border-slate-800">{COLS.map(([k, , d]) => <td key={k} className="py-1.5 pr-3">{fmt(r[k], d)}</td>)}</tr>)}</tbody></table></div>
            <div className="mt-3 flex items-center justify-between text-xs text-slate-500"><span>{rows.length} records</span>
              <span className="flex items-center gap-2"><button className="rounded border px-2 py-1 disabled:opacity-40 dark:border-slate-700" disabled={page === 0} onClick={() => setPage(page - 1)}>Prev</button>
                Page {page + 1} / {pages}<button className="rounded border px-2 py-1 disabled:opacity-40 dark:border-slate-700" disabled={page + 1 >= pages} onClick={() => setPage(page + 1)}>Next</button></span></div></>)}</Async>
        </Card>
      </div>
    </div>
  )
}
