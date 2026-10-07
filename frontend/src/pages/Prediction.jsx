import { useMemo, useState } from 'react'
import { Activity, FlaskConical } from 'lucide-react'
import { Area, CartesianGrid, ComposedChart, Legend, Line, Tooltip, XAxis, YAxis } from 'recharts'
import { Badge, Button, Card, DemoBanner, ErrorBox, Field, Kpi, Loading, NumberInput, PageTitle, Select, fmt, useAsync } from '../components/ui.jsx'
import { InfoTip } from '../components/viz.jsx'
import { Chart, axisProps, gridProps, tipProps } from '../charts/common.jsx'
import { useApp } from '../context/AppContext.jsx'
import { api } from '../services/api.js'

const Chip = (p) => <button type="button" {...p} className="rounded-full border border-slate-300 px-2.5 py-0.5 text-[11px] font-medium hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800" />

export default function Prediction() {
  const { dataset, dsId, battery } = useApp()
  const [cycle, setCycle] = useState('')
  const [horizon, setHorizon] = useState(dataset.headline_horizon)
  const [eol, setEol] = useState(dataset.eol_threshold)
  const [whatIf, setWhatIf] = useState(false)
  const [sc, setSc] = useState({ temp: 0, cur: 1 })
  const [state, setState] = useState({ loading: false, error: null, data: null })
  const hist = useAsync(() => api.history(battery, dsId), [battery, dsId])
  const lim = useAsync(() => api.scenarioLimits(battery, dsId, cycle ? Number(cycle) : undefined).catch(() => null), [battery, dsId, cycle])
  const L = lim.data

  const run = async () => {
    setState({ loading: true, error: null, data: null })
    const base = { battery_id: battery, dataset: dsId, horizon, ...(cycle ? { cycle: Number(cycle) } : {}) }
    const scen = whatIf ? { temp_shift_c: sc.temp || undefined, current_scale: sc.cur !== 1 ? sc.cur : undefined } : {}
    const isScen = scen.temp_shift_c || scen.current_scale
    try {
      const [fan, soh, deg, rul, baseFan] = await Promise.all([
        Promise.all(dataset.horizons.map((hh) => api.predictSoh({ ...base, horizon: hh, ...scen }))),
        api.predictSoh({ ...base, ...scen }),
        api.predictDeg({ ...base, ...scen }),
        api.predictRul({ ...base, ...scen, eol_threshold: Number(eol) }),
        isScen ? Promise.all(dataset.horizons.map((hh) => api.predictSoh({ ...base, horizon: hh }))) : Promise.resolve(null),
      ])
      const baseline = baseFan ? baseFan.find((f) => f.horizon === (horizon || dataset.headline_horizon)) : null
      setState({ loading: false, error: null, data: { soh, deg, rul, baseline, fan, baseFan, scen, isScen } })
    } catch (error) { setState({ loading: false, error, data: null }) }
  }

  const chart = useMemo(() => {
    const d = state.data; if (!d || !hist.data) return []
    const m = new Map(hist.data.records.filter((r) => r.cycle <= d.soh.from_cycle).map((r) => [r.cycle, { cycle: r.cycle, soh: r.soh }]))
    const set = (c, o) => m.set(c, { ...(m.get(c) || { cycle: c }), ...o })
    set(d.soh.from_cycle, { fan: d.soh.current_soh, baseline: d.soh.current_soh, band: [d.soh.current_soh, d.soh.current_soh] })
    d.fan.forEach((f) => set(f.target_cycle, { fan: f.predicted_soh, band: [f.interval_low, f.interval_high] }))
    if (d.baseFan) {
      d.baseFan.forEach((f) => set(f.target_cycle, { baseline: f.predicted_soh }))
    }
    return [...m.values()].sort((a, b) => a.cycle - b.cycle)
  }, [state.data, hist.data])

  const d = state.data
  const dLoss = d?.baseline ? d.soh.predicted_soh - d.baseline.predicted_soh : null
  const clampTo = (v, r) => (r ? Math.min(r[1], Math.max(r[0], v)) : v)
  return (
    <div className="mx-auto max-w-6xl">
      <PageTitle title="Prediction" subtitle="Forecast future SOH, degradation rate and RUL from the selected cycle" />
      <DemoBanner dataset={dataset} />
      <div className="grid gap-4 lg:grid-cols-3">
        <Card title="Inputs" subtitle={`Battery ${battery}`}>
          <div className="space-y-3">
            <Field label="From cycle" hint="blank = latest usable cycle"><NumberInput min={1} value={cycle} onChange={setCycle} /></Field>
            <Field label="Prediction horizon"><Select value={horizon} onChange={(v) => setHorizon(Number(v))} options={dataset.horizons.map((h) => ({ value: h, label: `${h} cycles ahead` }))} /></Field>
            <Field label="End-of-life SOH threshold (%)" hint={`ML RUL model was trained at ${dataset.eol_threshold}%; other values use extrapolation only.`}><NumberInput min={1} max={99} value={eol} onChange={setEol} /></Field>
            <label className="flex items-center gap-2 text-xs font-medium text-slate-600 dark:text-slate-300"><input type="checkbox" checked={whatIf} onChange={(e) => setWhatIf(e.target.checked)} /> What-if scenario</label>
            {whatIf && (
              <div className="space-y-3 rounded-lg border border-dashed border-slate-300 p-3 dark:border-slate-700">
                <p className="flex gap-1.5 text-[11px] text-slate-500"><FlaskConical className="h-3.5 w-3.5 shrink-0" /> Model-based scenario, not a physical simulation. All temperature (or current) features move together and stay inside the training data’s range.</p>
                {!L ? <p className="text-xs text-slate-500">Scenario limits unavailable for this cycle.</p> : (<>
                  <div className="flex flex-wrap gap-1.5">
                    <Chip onClick={() => setSc({ temp: L.temp_shift_c?.[0] ?? 0, cur: L.current_scale?.[0] ?? 1 })}>Cooler &amp; gentler</Chip>
                    <Chip onClick={() => setSc({ temp: L.temp_shift_c?.[1] ?? 0, cur: L.current_scale?.[1] ?? 1 })}>Hotter &amp; heavier</Chip>
                    <Chip onClick={() => setSc({ temp: 0, cur: 1 })}>Reset</Chip></div>
                  {L.temp_shift_c && <Field label={`Temperature shift: ${sc.temp >= 0 ? '+' : ''}${fmt(clampTo(sc.temp, L.temp_shift_c), 1)} °C`} hint={`allowed ${fmt(L.temp_shift_c[0], 1)} to +${fmt(L.temp_shift_c[1], 1)} °C`}>
                    <input type="range" className="w-full" min={L.temp_shift_c[0]} max={L.temp_shift_c[1]} step={0.1} value={clampTo(sc.temp, L.temp_shift_c)} onChange={(e) => setSc((o) => ({ ...o, temp: Number(e.target.value) }))} /></Field>}
                  {L.current_scale && <Field label={`Discharge current: ${fmt(clampTo(sc.cur, L.current_scale) * 100, 0)} % of actual`} hint={`allowed ${fmt(L.current_scale[0] * 100, 0)}–${fmt(L.current_scale[1] * 100, 0)} %`}>
                    <input type="range" className="w-full" min={L.current_scale[0]} max={L.current_scale[1]} step={0.01} value={clampTo(sc.cur, L.current_scale)} onChange={(e) => setSc((o) => ({ ...o, cur: Number(e.target.value) }))} /></Field>}
                </>)}
              </div>)}
            <Button onClick={run} disabled={state.loading}><Activity className="h-4 w-4" /> Run prediction</Button>
          </div>
        </Card>

        <div className="space-y-4 lg:col-span-2">
          {state.loading && <Loading label="Running models…" />}
          {state.error && <ErrorBox error={state.error} />}
          {!d && !state.loading && !state.error && <Card><div className="py-10 text-center text-sm text-slate-500">Choose inputs and press <b>Run prediction</b>.</div></Card>}
          {d && (<>
            <div className="grid gap-3 sm:grid-cols-3">
              <Kpi label={`SOH at cycle ${d.soh.target_cycle}`} value={fmt(d.soh.predicted_soh, 1)} unit="%" kind="predicted" hint={dLoss != null ? `${dLoss >= 0 ? '+' : ''}${fmt(dLoss, 2)} pp vs baseline` : `band ${fmt(d.soh.interval_low, 1)}–${fmt(d.soh.interval_high, 1)}`} />
              <Kpi label="Degradation rate" value={fmt(d.deg.predicted_rate_pct_per_cycle, 3)} unit="% / cycle" kind="predicted" hint={`observed ${fmt(d.deg.observed_rolling_rate_25, 3)}`} />
              <Kpi label="Estimated RUL" value={d.rul.ml_rul_cycles != null ? fmt(d.rul.ml_rul_cycles, 0) : fmt(d.rul.extrapolation_rul_cycles, 0)} unit="cycles" kind="predicted" hint={d.rul.ml_rul_cycles != null ? `ML · extrapolation ${fmt(d.rul.extrapolation_rul_cycles, 0)}` : 'extrapolation only'} />
            </div>
            <p className="text-xs text-slate-500">{d.rul.statement} {d.rul.ml_note && <span className="text-amber-600">{d.rul.ml_note}</span>} {d.isScen && <Badge tone="warn">what-if · stress {fmt(d.soh.stress_multiplier, 2)}×</Badge>} {d.soh.monotonic_constraint_applied && <Badge tone="neutral">forecast capped: SOH cannot rise</Badge>}{' '}
              Current SOH {fmt(d.soh.current_soh, 1)}% at cycle {d.soh.from_cycle} (derived). Model: {d.soh.model}.{d.isScen && <span className="text-amber-600"> Scenario: {d.scen.temp_shift_c ? `${d.scen.temp_shift_c >= 0 ? '+' : ''}${fmt(d.scen.temp_shift_c, 1)} °C` : ''}{d.scen.temp_shift_c && d.scen.current_scale ? ', ' : ''}{d.scen.current_scale ? `${fmt(d.scen.current_scale * 100, 0)}% current` : ''}</span>}</p>
            <Card title="Forecast fan" subtitle="Solid = measured/derived history · dashed = ML forecast at every trained horizon · shaded = 5–95% residual band"
              right={<InfoTip text="Band = empirical 5–95% quantiles of out-of-fold residuals from held-out batteries. It is marginal and coarse — not a guaranteed coverage interval." />}>
              <Chart height={300}><ComposedChart data={chart}><CartesianGrid {...gridProps} /><XAxis dataKey="cycle" type="number" domain={['dataMin', 'dataMax']} {...axisProps} /><YAxis domain={['auto', 'auto']} {...axisProps} unit="%" />
                <Tooltip {...tipProps} formatter={(v) => (Array.isArray(v) ? `${fmt(v[0], 1)} – ${fmt(v[1], 1)} %` : fmt(v, 2) + ' %')} /><Legend wrapperStyle={{ fontSize: 11 }} />
                <Area dataKey="band" name="Forecast band" stroke="none" fill="#f59e0b" fillOpacity={0.18} connectNulls />
                <Line dataKey="soh" name="History" stroke="#14b8a6" dot={false} strokeWidth={2} connectNulls />
                <Line dataKey="fan" name={d.isScen ? 'Forecast (what-if)' : 'Forecast'} stroke="#f59e0b" strokeDasharray="5 3" dot={{ r: 4 }} connectNulls />
                {d.baseline && <Line dataKey="baseline" name="Forecast (baseline)" stroke="#94a3b8" strokeDasharray="5 3" dot={{ r: 3 }} connectNulls />}</ComposedChart></Chart>
            </Card></>)}
        </div>
      </div>
    </div>
  )
}
