import { useMemo, useState } from 'react'
import { Legend, Line, LineChart, Bar, BarChart, CartesianGrid, Cell, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis, ReferenceLine } from 'recharts'
import { Async, Badge, Card, DemoBanner, Field, PageTitle, Select, fmt, useAsync } from '../components/ui.jsx'
import { Chart, axisProps, gridProps, tipProps } from '../charts/common.jsx'
import { useApp } from '../context/AppContext.jsx'
import { api } from '../services/api.js'

export default function ModelPerformance() {
  const { dataset, dsId } = useApp()
  const [h, setH] = useState(dataset.headline_horizon)
  const perf = useAsync(() => api.performance(dsId, h), [dsId, h])
  const models = useAsync(() => api.models(dsId), [dsId])
  const curve = useAsync(() => api.horizonCurve(dsId), [dsId])
  const pts = useMemo(() => (perf.data?.points || []).filter((_, i) => i % 2 === 0).map((p) => ({ ...p, res: p.y_true - p.y_pred })), [perf.data])

  return (
    <div className="mx-auto max-w-6xl">
      <PageTitle title="Model Performance" subtitle="Leave-one-battery-out: every score is on a battery the model never saw" />
      <DemoBanner dataset={dataset} />
      <Async state={perf}>{(p) => {
        const names = Object.keys(p.forecast)
        const bars = names.map((n) => ({ name: n, rmse: p.forecast[n].rmse, trained: p.forecast[n].trained }))
        return (<div className="grid gap-4 lg:grid-cols-2">
          <Card className="lg:col-span-2" title={`SOH forecast comparison — ${p.horizon} cycles ahead`} right={<div className="w-36"><Field label="Horizon"><Select value={h} onChange={(v) => setH(Number(v))} options={dataset.horizons.map((x) => ({ value: x, label: `${x} cycles` }))} /></Field></div>}>
            <div className="overflow-x-auto"><table className="w-full text-left text-sm"><thead className="text-xs text-slate-500"><tr><th className="py-1">Model</th><th>MAE (pp)</th><th>RMSE (pp)</th><th>R²</th><th title="R² of the SOH change, i.e. skill beyond the trivially predictable current level">R² (Δ)</th></tr></thead>
              <tbody>{names.map((n) => { const m = p.forecast[n]; return (
                <tr key={n} className="num border-t border-slate-100 dark:border-slate-800"><td className="py-1.5 font-medium">{n} {n === p.best_model && <Badge tone="good">best</Badge>} {!m.trained && <Badge>reference, not a model</Badge>}</td>
                  <td>{fmt(m.mae, 3)}</td><td>{fmt(m.rmse, 3)}</td><td>{fmt(m.r2, 4)}</td><td>{fmt(m.r2_delta, 3)}</td></tr>)})}</tbody></table></div>
            <p className="mt-2 text-xs text-slate-500">Only models actually trained are listed. R² on absolute SOH is inflated because SOH is highly autocorrelated; compare against the no-model references and R² (Δ).
              {models.data?.model_selection_note && <> {models.data.model_selection_note}</>}</p>
          </Card>
          <Card className="lg:col-span-2" title="Error vs forecast horizon" subtitle="MAE (percentage points of SOH), leave-one-battery-out. Models should stay below the grey no-model references.">
            <Async state={curve}>{(c) => {
              const names = Object.keys(c.horizons[0].models)
              const data = c.horizons.map((h) => ({ horizon: h.horizon, ...Object.fromEntries(names.map((n) => [n, h.models[n].mae])) }))
              const col = { linear_regression: '#14b8a6', random_forest: '#6366f1', xgboost: '#f59e0b', hist_gradient_boosting: '#f59e0b', 'persistence (no model)': '#94a3b8', 'linear extrapolation (no model)': '#cbd5e1' }
              return <Chart height={260}><LineChart data={data}><CartesianGrid {...gridProps} /><XAxis dataKey="horizon" {...axisProps} unit=" cyc" /><YAxis {...axisProps} scale="sqrt" domain={[0, 'auto']} />
                <Tooltip {...tipProps} formatter={(v) => fmt(v, 3) + ' pp'} /><Legend wrapperStyle={{ fontSize: 11 }} />
                {names.map((n) => <Line key={n} dataKey={n} stroke={col[n] || '#64748b'} strokeWidth={c.horizons[0].models[n].trained ? 2.2 : 1.4} strokeDasharray={c.horizons[0].models[n].trained ? undefined : '4 3'} dot={{ r: 3 }} />)}</LineChart></Chart>
            }}</Async>
          </Card>
          <Card title="RMSE by model" subtitle="lower is better; grey = no-model reference">
            <Chart height={240}><BarChart data={bars} layout="vertical" margin={{ left: 70 }}><CartesianGrid {...gridProps} horizontal={false} /><XAxis type="number" {...axisProps} /><YAxis type="category" dataKey="name" width={170} {...axisProps} />
              <Tooltip {...tipProps} formatter={(v) => fmt(v, 3) + ' pp'} /><Bar dataKey="rmse" radius={3}>{bars.map((b) => <Cell key={b.name} fill={b.trained ? '#14b8a6' : '#94a3b8'} />)}</Bar></BarChart></Chart>
          </Card>
          <Card title="Validation" subtitle="how leakage is prevented"><p className="text-xs leading-relaxed text-slate-600 dark:text-slate-300">{p.validation}</p>
            {models.data && <p className="mt-2 text-xs text-slate-500">Python {models.data.versions.python} · scikit-learn {models.data.versions.sklearn}</p>}</Card>
          <Card title="Actual vs predicted SOH" subtitle={`${p.best_model}, held-out batteries`}>
            <Chart height={280}><ScatterChart><CartesianGrid {...gridProps} /><XAxis type="number" dataKey="y_true" name="Actual" unit="%" {...axisProps} domain={['auto', 'auto']} /><YAxis type="number" dataKey="y_pred" name="Predicted" unit="%" {...axisProps} domain={['auto', 'auto']} />
              <ZAxis range={[14, 14]} /><Tooltip {...tipProps} formatter={(v) => fmt(v, 2)} /><ReferenceLine segment={[{ x: 40, y: 40 }, { x: 105, y: 105 }]} stroke="#94a3b8" strokeDasharray="4 4" />
              <Scatter data={pts} fill="#14b8a6" fillOpacity={0.55} /></ScatterChart></Chart>
          </Card>
          <Card title="Residuals" subtitle="actual − predicted vs predicted (percentage points)">
            <Chart height={280}><ScatterChart><CartesianGrid {...gridProps} /><XAxis type="number" dataKey="y_pred" name="Predicted" unit="%" {...axisProps} domain={['auto', 'auto']} /><YAxis type="number" dataKey="res" name="Residual" {...axisProps} />
              <ZAxis range={[14, 14]} /><Tooltip {...tipProps} formatter={(v) => fmt(v, 2)} /><ReferenceLine y={0} stroke="#94a3b8" /><Scatter data={pts} fill="#6366f1" fillOpacity={0.55} /></ScatterChart></Chart>
          </Card>
          <Card title="Per-battery error" subtitle={`${p.best_model}; spread across batteries shows generalisation risk`}>
            <table className="w-full text-left text-xs"><thead className="text-slate-500"><tr><th className="py-1">Battery</th><th>MAE</th><th>RMSE</th><th>n</th></tr></thead>
              <tbody>{Object.entries(p.per_battery_best).map(([b, m]) => <tr key={b} className="num border-t border-slate-100 dark:border-slate-800"><td className="py-1.5 font-medium">{b}</td><td>{fmt(m.mae, 3)}</td><td>{fmt(m.rmse, 3)}</td><td>{m.n}</td></tr>)}</tbody></table>
          </Card>
          <Card title="RUL estimation error (cycles)" subtitle="ground truth = cycles until observed SOH first ≤ threshold; cells that never reach it are excluded">
            {p.rul?.metrics ? <table className="w-full text-left text-xs"><thead className="text-slate-500"><tr><th className="py-1">Estimator</th><th>MAE</th><th>RMSE</th><th>MAE (RUL ≤ 50)</th></tr></thead>
              <tbody>{Object.entries(p.rul.metrics).map(([n, m]) => <tr key={n} className="num border-t border-slate-100 dark:border-slate-800"><td className="py-1.5 font-medium">{n} {n === p.rul.best_model && <Badge tone="good">best</Badge>}</td><td>{fmt(m.mae, 2)}</td><td>{fmt(m.rmse, 2)}</td><td>{fmt(m.mae_when_true_rul_le_50, 2)}</td></tr>)}</tbody></table>
              : <p className="text-xs text-slate-500">RUL was not evaluated (too few batteries reach the end-of-life threshold).</p>}
          </Card>
        </div>)
      }}</Async>
    </div>
  )
}
