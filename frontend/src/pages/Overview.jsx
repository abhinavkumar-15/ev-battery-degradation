import { useState } from 'react'
import { Gauge as GaugeIcon, Hourglass, Layers, Thermometer, TrendingDown } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, Line, LineChart, ReferenceLine, Tooltip, XAxis, YAxis } from 'recharts'
import { Async, Badge, Card, DemoBanner, Field, NumberInput, PageTitle, fmt, useAsync } from '../components/ui.jsx'
import { Bar2, Gauge, Sparkline, healthColor, statusTone } from '../components/viz.jsx'
import { Chart, COLORS, axisProps, gridProps, tipProps } from '../charts/common.jsx'
import { useApp } from '../context/AppContext.jsx'
import { api } from '../services/api.js'

const Stat = ({ icon: Icon, label, value, unit, kind, hint, spark, sparkKey, color }) => (
  <div className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
    <div className="flex items-center justify-between text-xs font-medium text-slate-500 dark:text-slate-400"><span>{label}</span><Icon className="h-4 w-4 text-slate-400" aria-hidden /></div>
    <div className="mt-1.5 flex items-baseline gap-1"><span className="num text-2xl font-semibold text-slate-900 dark:text-white">{value}</span>{unit && <span className="text-xs text-slate-500">{unit}</span>}</div>
    {spark && <Sparkline data={spark} dataKey={sparkKey} color={color} />}
    <div className="mt-1.5 flex items-center gap-2">{kind && <Badge tone={kind}>{kind}</Badge>}{hint && <span className="text-[11px] text-slate-500">{hint}</span>}</div>
  </div>
)

export default function Overview() {
  const { dataset, dsId, battery, setBattery } = useApp()
  const [asOf, setAsOf] = useState('')
  const cyc = asOf === '' ? undefined : Number(asOf)
  const sum = useAsync(() => api.summary(battery, dsId), [battery, dsId])
  const hist = useAsync(() => api.history(battery, dsId), [battery, dsId])
  const fleet = useAsync(() => api.fleet(dsId), [dsId])
  const rul = useAsync(() => api.predictRul({ battery_id: battery, dataset: dsId, ...(cyc ? { cycle: cyc } : {}) }), [battery, dsId, cyc])
  const exp = useAsync(() => api.explainBattery(battery, dsId, cyc), [battery, dsId, cyc])
  const rec = cyc ? hist.data?.records.find((r) => r.cycle === cyc) : null
  const tail = (hist.data?.records || []).filter((r) => !cyc || r.cycle <= cyc).slice(-40)

  return (
    <div className="mx-auto max-w-6xl">
      <div className="flex items-start justify-between gap-3">
        <PageTitle title="Overview" subtitle={`Battery ${battery} · dataset ${dsId}`} />
        <div className="w-32"><Field label="As of cycle" hint="blank = latest"><NumberInput min={1} value={asOf} onChange={setAsOf} /></Field></div>
      </div>
      <DemoBanner dataset={dataset} />
      <Async state={sum}>{(s) => {
        const soh = rec ? rec.soh : s.current_soh
        const eol = s.eol_threshold
        const status = rec ? (soh <= eol ? 'End of life reached' : soh >= eol + 10 ? 'Healthy' : 'Ageing - approaching EOL') : s.status
        return (<>
          <div className="grid gap-4 lg:grid-cols-[260px_1fr]">
            <Card className="flex flex-col items-center justify-center gap-2 text-center">
              <Gauge value={soh} eol={eol} />
              <Badge tone={statusTone(status)}>{status}</Badge>
              <p className="text-[11px] text-slate-500">Red tick = end-of-life threshold ({eol}% SOH)</p>
              <Async state={rul} empty={null}>{(r) => {
                const rulv = r.ml_rul_cycles ?? r.extrapolation_rul_cycles
                const used = r.from_cycle
                return (<div className="w-full pt-1 text-left"><div className="mb-1 flex justify-between text-[11px] text-slate-500"><span>Cycle {used}</span><span>est. life ≈ {fmt(used + rulv, 0)}</span></div>
                  <Bar2 value={used} max={used + rulv || 1} color={healthColor(soh, eol)} /></div>)
              }}</Async>
            </Card>
            <div className="grid content-start gap-3 sm:grid-cols-2">
              <Stat icon={TrendingDown} label="Degradation rate" value={fmt(rec ? rec.rolling_degradation_rate_25 : s.degradation_rate_pct_per_cycle, 3)} unit="% / cycle" kind="derived" hint="trailing 25-cycle mean" spark={tail} sparkKey="rolling_degradation_rate_10" color="#6366f1" />
              <Async state={rul} empty={null}>{(r) => (
                <Stat icon={Hourglass} label="Estimated RUL" value={fmt(r.ml_rul_cycles ?? r.extrapolation_rul_cycles, 0)} unit="cycles" kind="predicted" hint={r.ml_rul_cycles != null ? `ML (${r.ml_model}) · extrapolation ${fmt(r.extrapolation_rul_cycles, 0)}` : 'extrapolation only'} />)}</Async>
              <Stat icon={GaugeIcon} label="Capacity" value={fmt(rec ? rec.capacity_ah : s.current_capacity_ah, 3)} unit="Ah" kind="measured" hint={`reference ${fmt(s.reference_capacity_ah, 1)} Ah`} spark={tail} sparkKey="capacity_ah" color="#14b8a6" />
              <Stat icon={Thermometer} label="Mean discharge temp" value={fmt(rec ? rec.dis_t_mean : tail.at(-1)?.dis_t_mean, 1)} unit="°C" kind="measured" hint={`${s.cycles} cycles observed`} spark={tail} sparkKey="dis_t_mean" color="#f59e0b" />
            </div>
          </div>
          <p className="mt-2 text-xs text-slate-500">RUL is estimated using an {eol}% SOH end-of-life threshold. It is an ML estimate, not an exact physical prediction.
            {s.eol_cycle_observed && <> This cell crossed the threshold at cycle {s.eol_cycle_observed}; for cycles after that, RUL is 0 by definition — use “As of cycle” to look earlier in its life.</>}</p>

          <div className="mt-4 grid gap-4 lg:grid-cols-3">
            <Card className="lg:col-span-2" title="SOH trajectory" subtitle="Derived from measured capacity ÷ reference capacity">
              <Async state={hist}>{(h) => (
                <Chart><LineChart data={h.records}><CartesianGrid {...gridProps} /><XAxis dataKey="cycle" {...axisProps} /><YAxis domain={['auto', 'auto']} {...axisProps} unit="%" />
                  <Tooltip {...tipProps} formatter={(v) => fmt(v, 2) + ' %'} labelFormatter={(l) => `cycle ${l}`} />
                  <ReferenceLine y={eol} stroke="#ef4444" strokeDasharray="4 4" label={{ value: `EOL ${eol}%`, fill: '#ef4444', fontSize: 11, position: 'insideTopRight' }} />
                  {cyc && <ReferenceLine x={cyc} stroke="#6366f1" strokeDasharray="3 3" />}
                  <Line dataKey="soh" stroke="#14b8a6" dot={false} strokeWidth={2} name="SOH" /></LineChart></Chart>)}</Async>
            </Card>
            <Card title="Factors associated with the forecast" subtitle="Local SHAP · model association, not causation">
              <Async state={exp}>{(e) => (
                <Chart height={250}><BarChart data={e.contributions.slice(0, 6)} layout="vertical" margin={{ left: 40 }}>
                  <CartesianGrid {...gridProps} horizontal={false} /><XAxis type="number" {...axisProps} /><YAxis type="category" dataKey="feature" width={130} {...axisProps} />
                  <Tooltip {...tipProps} formatter={(v) => fmt(v, 3) + ' pp'} /><Bar dataKey="shap" fill="#6366f1" radius={3} name="SHAP" /></BarChart></Chart>)}</Async>
            </Card>
          </div>

          <Card className="mt-4" title="Fleet comparison" subtitle="Every cell in this dataset — click a row or line to select it">
            <Async state={fleet}>{(f) => {
              const maxLen = Math.max(...f.cells.map((c) => c.cycles))
              const byCycle = new Map()
              f.cells.forEach((c) => c.series.forEach((p) => byCycle.set(p.cycle, { ...(byCycle.get(p.cycle) || { cycle: p.cycle }), [c.id]: p.soh })))
              const data = [...byCycle.values()].sort((a, b) => a.cycle - b.cycle)
              return (<div className="grid gap-4 lg:grid-cols-2">
                <Chart height={300}><LineChart data={data}><CartesianGrid {...gridProps} /><XAxis dataKey="cycle" type="number" domain={[1, maxLen]} {...axisProps} /><YAxis domain={['auto', 'auto']} {...axisProps} unit="%" />
                  <Tooltip {...tipProps} formatter={(v) => fmt(v, 1) + ' %'} /><ReferenceLine y={f.eol_threshold} stroke="#ef4444" strokeDasharray="4 4" />
                  {f.cells.map((c, i) => <Line key={c.id} dataKey={c.id} dot={false} connectNulls strokeWidth={c.id === battery ? 3 : 1.1} strokeOpacity={c.id === battery ? 1 : 0.45}
                    stroke={c.id === battery ? '#14b8a6' : COLORS[i % COLORS.length]} onClick={() => setBattery(c.id)} style={{ cursor: 'pointer' }} />)}</LineChart></Chart>
                <div className="max-h-[300px] overflow-auto"><table className="w-full text-left text-xs"><thead className="sticky top-0 bg-white text-slate-500 dark:bg-slate-900"><tr><th className="py-1">Cell</th><th>Status</th><th>Final SOH</th><th>EOL cycle</th><th>Amb °C</th><th>I (A)</th></tr></thead>
                  <tbody>{f.cells.map((c) => (
                    <tr key={c.id} onClick={() => setBattery(c.id)} className={`cursor-pointer border-t border-slate-100 dark:border-slate-800 ${c.id === battery ? 'bg-brand-50 dark:bg-brand-700/20' : 'hover:bg-slate-50 dark:hover:bg-slate-800/50'}`}>
                      <td className="py-1.5 font-medium">{c.id}</td><td><Badge tone={statusTone(c.status)}>{c.status.replace(' reached', '').replace(' - approaching EOL', '')}</Badge></td>
                      <td className="num w-28"><div className="mb-0.5">{fmt(c.final_soh, 1)}%</div><Bar2 value={c.final_soh} color={healthColor(c.final_soh, f.eol_threshold)} /></td>
                      <td className="num">{c.eol_cycle ?? '—'}</td><td className="num">{fmt(c.ambient_temp_c, 0)}</td><td className="num">{fmt(c.mean_discharge_current_a, 1)}</td></tr>))}</tbody></table></div>
              </div>)
            }}</Async>
          </Card>
        </>)
      }}</Async>
    </div>
  )
}
