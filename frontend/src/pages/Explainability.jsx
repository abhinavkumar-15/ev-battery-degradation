import { useMemo, useState } from 'react'
import { Info } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, Cell, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from 'recharts'
import { Async, Card, DemoBanner, Field, NumberInput, PageTitle, Select, fmt, useAsync } from '../components/ui.jsx'
import { Chart, axisProps, gridProps, tipProps } from '../charts/common.jsx'
import { useApp } from '../context/AppContext.jsx'
import { api } from '../services/api.js'

export default function Explainability() {
  const { dataset, dsId, battery } = useApp()
  const [cycle, setCycle] = useState('')
  const [feat, setFeat] = useState(null)
  const g = useAsync(() => api.explain(dsId), [dsId])
  const l = useAsync(() => api.explainBattery(battery, dsId, cycle || undefined), [battery, dsId, cycle])

  const models = useAsync(() => api.models(dsId), [dsId])
  const dir = useMemo(() => {
    const sm = g.data?.sample; if (!sm) return {}
    const out = {}
    g.data.features.forEach((f, i) => {
      const x = sm.values.map((r) => r[i]), y = sm.shap.map((r) => r[i])
      const mx = x.reduce((a, b) => a + b, 0) / x.length, my = y.reduce((a, b) => a + b, 0) / y.length
      const sxy = x.reduce((a, v, k) => a + (v - mx) * (y[k] - my), 0), sxx = x.reduce((a, v) => a + (v - mx) ** 2, 0), syy = y.reduce((a, v) => a + (v - my) ** 2, 0)
      out[f] = sxx > 0 && syy > 0 ? sxy / Math.sqrt(sxx * syy) : 0
    })
    return out
  }, [g.data])
  const dep = useMemo(() => {
    const s = g.data?.sample; if (!s || !feat) return []
    const i = g.data.features.indexOf(feat)
    return s.shap.map((row, k) => ({ x: s.values[k][i], y: row[i] }))
  }, [g.data, feat])

  return (
    <div className="mx-auto max-w-6xl">
      <PageTitle title="Explainability" subtitle="Which features does the model associate most with its SOH forecast?" />
      <DemoBanner dataset={dataset} />
      <div className="mb-4 flex gap-2 rounded-lg border border-sky-200 bg-sky-50 p-3 text-xs text-sky-900 dark:border-sky-900 dark:bg-sky-950 dark:text-sky-200">
        <Info className="mt-0.5 h-4 w-4 shrink-0" />
        <div>SHAP values describe how the <b>fitted model</b> uses each feature. They are <b>associations, not causes</b>: features such as temperature, current and cycle count are correlated,
          and the data are observational. Establishing causation requires controlled experiments.</div>
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Global importance" subtitle={g.data ? `mean |SHAP| · method: ${g.data.method}` : ''}>
          <Async state={g}>{(d) => {
            const top = d.importance.slice(0, 15)
            return (<><Chart height={400}><BarChart data={top} layout="vertical" margin={{ left: 50 }}><CartesianGrid {...gridProps} horizontal={false} /><XAxis type="number" {...axisProps} />
              <YAxis type="category" dataKey="feature" width={150} {...axisProps} /><Tooltip {...tipProps} formatter={(v) => fmt(v, 3)} labelFormatter={(f) => top.find((t) => t.feature === f)?.doc || f} />
              <Bar dataKey="mean_abs_shap" name="mean |SHAP|" radius={3} onClick={(e) => setFeat(e.feature)}>{top.map((t) => <Cell key={t.feature} fill={t.feature === feat ? '#f59e0b' : '#14b8a6'} cursor="pointer" />)}</Bar></BarChart></Chart>
              <p className="mt-2 text-xs text-slate-500">{d.interpretation} Click a bar for its dependence plot.</p></>)
          }}</Async>
        </Card>
        <div className="space-y-4">
          <Card title="How the model uses the top features" subtitle="Direction of association in the sampled rows (correlation of feature value with its SHAP value)">
            <Async state={g}>{(d) => (<table className="w-full text-left text-xs"><thead className="text-slate-500"><tr><th className="py-1">Feature</th><th>Higher value →</th><th>Meaning</th></tr></thead>
              <tbody>{d.importance.slice(0, 7).map((r) => { const c = dir[r.feature] ?? 0; return (
                <tr key={r.feature} className="border-t border-slate-100 dark:border-slate-800"><td className="py-1.5 font-mono">{r.feature}</td>
                  <td className={Math.abs(c) < 0.2 ? 'text-slate-500' : c > 0 ? 'text-emerald-600' : 'text-rose-500'}>{Math.abs(c) < 0.2 ? 'mixed' : c > 0 ? '↑ smaller loss' : '↓ larger loss'} <span className="num text-slate-400">({fmt(c, 2)})</span></td>
                  <td className="text-slate-500">{r.doc}</td></tr>)})}</tbody></table>)}</Async>
            {models.data && Object.keys(models.data.collinear_dropped || {}).length > 0 && <p className="mt-2 text-[11px] text-slate-500">Exact duplicates removed before training: {Object.entries(models.data.collinear_dropped).map(([k, v]) => `${k} (≡ ${v})`).join(', ')}.
              Remaining correlated features (e.g. ambient and rolling temperature) can receive offsetting credit — another reason SHAP is not causal evidence.</p>}
          </Card>
          <Card title={`Local explanation — ${battery}`} subtitle="Contribution of each feature to the predicted SOH change (percentage points)"
            right={<div className="w-28"><Field label="Cycle"><NumberInput min={1} value={cycle} onChange={setCycle} placeholder="latest" /></Field></div>}>
            <Async state={l}>{(d) => (<>
              <Chart height={300}><BarChart data={d.contributions} layout="vertical" margin={{ left: 50 }}><CartesianGrid {...gridProps} horizontal={false} /><XAxis type="number" {...axisProps} />
                <YAxis type="category" dataKey="feature" width={150} {...axisProps} /><Tooltip {...tipProps} formatter={(v) => fmt(v, 3) + ' pp'} />
                <Bar dataKey="shap" radius={3}>{d.contributions.map((c) => <Cell key={c.feature} fill={c.shap >= 0 ? '#14b8a6' : '#ef4444'} />)}</Bar></BarChart></Chart>
              <p className="mt-2 text-xs text-slate-500">Baseline {fmt(d.base_value, 2)} pp → prediction {fmt(d.prediction_delta, 2)} pp over {d.horizon} cycles (cycle {d.from_cycle}). {d.interpretation}</p></>)}</Async>
          </Card>
          <Card title={feat ? `Dependence: ${feat}` : 'Dependence plot'} subtitle="Each dot is a sampled training row: feature value vs its SHAP contribution">
            {feat ? <Chart height={220}><ScatterChart><CartesianGrid {...gridProps} /><XAxis type="number" dataKey="x" name={feat} {...axisProps} domain={['auto', 'auto']} /><YAxis type="number" dataKey="y" name="SHAP" {...axisProps} />
              <ZAxis range={[18, 18]} /><Tooltip {...tipProps} cursor={{ strokeDasharray: '3 3' }} formatter={(v) => fmt(v, 3)} /><Scatter data={dep} fill="#6366f1" fillOpacity={0.6} /></ScatterChart></Chart>
              : <p className="py-8 text-center text-xs text-slate-500">Select a feature in the global chart.</p>}
          </Card>
        </div>
      </div>
    </div>
  )
}
