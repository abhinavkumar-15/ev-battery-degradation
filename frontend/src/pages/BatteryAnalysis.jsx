import { useMemo, useState } from 'react'
import { Brush, CartesianGrid, Legend, Line, LineChart, ReferenceLine, Tooltip, XAxis, YAxis } from 'recharts'
import { Async, Card, DemoBanner, Empty, Field, NumberInput, PageTitle, Select, fmt, useAsync } from '../components/ui.jsx'
import { InfoTip, Segmented } from '../components/viz.jsx'
import { Chart, axisProps, gridProps, tipProps } from '../charts/common.jsx'
import { useApp } from '../context/AppContext.jsx'
import { api } from '../services/api.js'

export default function BatteryAnalysis() {
  const { dataset, dsId, battery, batteries } = useApp()
  const [tab, setTab] = useState('health')
  const [range, setRange] = useState({ lo: 1, hi: '' })
  const [other, setOther] = useState('')
  const [horizon, setHorizon] = useState(dataset.headline_horizon)
  const hist = useAsync(() => api.history(battery, dsId), [battery, dsId])
  const cmp = useAsync(() => (other ? api.history(other, dsId) : Promise.resolve(null)), [other, dsId])
  const perf = useAsync(() => api.performance(dsId, horizon), [dsId, horizon])

  const rows = useMemo(() => {
    const base = (hist.data?.records || []).filter((r) => r.cycle >= (range.lo || 1) && (range.hi === '' || r.cycle <= range.hi))
    if (!cmp.data) return base
    const m = new Map(cmp.data.records.map((r) => [r.cycle, r]))
    return base.map((r) => ({ ...r, soh_other: m.get(r.cycle)?.soh, cap_other: m.get(r.cycle)?.capacity_ah }))
  }, [hist.data, cmp.data, range])
  const pts = useMemo(() => (perf.data?.points || []).filter((p) => String(p.battery_id) === String(battery)).map((p) => ({ ...p, target_cycle: p.cycle + horizon })), [perf.data, battery, horizon])
  const hasRes = rows.some((r) => r.re_ohm != null)
  const eol = dataset.eol_threshold

  const L = ({ keys, unit, h = 240, brush, eolLine, tipFmt = 3 }) => (
    <Chart height={h}><LineChart data={rows}><CartesianGrid {...gridProps} /><XAxis dataKey="cycle" {...axisProps} /><YAxis domain={['auto', 'auto']} {...axisProps} unit={unit} />
      <Tooltip {...tipProps} formatter={(v) => fmt(v, tipFmt)} labelFormatter={(l) => `cycle ${l}`} /><Legend wrapperStyle={{ fontSize: 11 }} />
      {eolLine && <ReferenceLine y={eol} stroke="#ef4444" strokeDasharray="4 4" />}
      {keys.map((k) => <Line key={k.key} dataKey={k.key} name={k.name} stroke={k.color} dot={false} strokeWidth={k.w || 1.8} strokeDasharray={k.dash} connectNulls />)}
      {brush && <Brush dataKey="cycle" height={18} stroke="var(--axis)" travellerWidth={8} />}</LineChart></Chart>)

  return (
    <div className="mx-auto max-w-6xl">
      <PageTitle title="Battery Analysis" subtitle={`Measured and derived history for ${battery}`} />
      <DemoBanner dataset={dataset} />
      <Card className="mb-4"><div className="grid gap-3 sm:grid-cols-4">
        <Field label="From cycle"><NumberInput min={1} value={range.lo} onChange={(v) => setRange((r) => ({ ...r, lo: v }))} /></Field>
        <Field label="To cycle" hint="blank = last"><NumberInput min={1} value={range.hi} onChange={(v) => setRange((r) => ({ ...r, hi: v }))} /></Field>
        <Field label="Compare with"><Select value={other} onChange={setOther} options={[{ value: '', label: '— none —' }, ...batteries.filter((b) => b.id !== battery).map((b) => b.id)]} /></Field>
        <Field label="Forecast horizon"><Select value={horizon} onChange={(v) => setHorizon(Number(v))} options={dataset.horizons.map((h) => ({ value: h, label: `${h} cycles` }))} /></Field>
      </div></Card>
      <div className="mb-3"><Segmented value={tab} onChange={setTab} options={[{ value: 'health', label: 'Health' }, { value: 'ops', label: 'Operating conditions' }, ...(hasRes ? [{ value: 'res', label: 'Resistance' }] : []), { value: 'fc', label: 'Held-out forecast' }]} /></div>
      <Async state={hist}>{() => rows.length === 0 ? <Empty title="No cycles in this range" hint="Widen the cycle range." /> : (<>
        {tab === 'health' && (<div className="grid gap-4 lg:grid-cols-2">
          <Card className="lg:col-span-2" title="SOH vs cycle" subtitle="derived · drag the slider below the chart to zoom" right={<InfoTip text="SOH = measured capacity ÷ reference capacity × 100. Red dashed line = end-of-life threshold." />}>
            <L h={290} brush eolLine tipFmt={2} keys={[{ key: 'soh', name: battery, color: '#14b8a6', w: 2.2 }, ...(other ? [{ key: 'soh_other', name: other, color: '#6366f1', dash: '5 3' }] : [])]} unit="%" /></Card>
          <Card title="Capacity (Ah)" subtitle="measured"><L keys={[{ key: 'capacity_ah', name: battery, color: '#14b8a6' }, ...(other ? [{ key: 'cap_other', name: other, color: '#6366f1', dash: '5 3' }] : [])]} /></Card>
          <Card title="Degradation rate (% SOH / cycle)" subtitle="derived · per-cycle values are noisy, rolling means smooth them"
            right={<InfoTip text="Rate = (SOH[t−1] − SOH[t]) ÷ cycle gap. The 10- and 25-cycle rolling means only look backwards." />}>
            <L keys={[{ key: 'rolling_degradation_rate_10', name: 'rolling 10', color: '#f59e0b' }, { key: 'rolling_degradation_rate_25', name: 'rolling 25', color: '#ef4444' }]} /></Card>
        </div>)}
        {tab === 'ops' && (<div className="grid gap-4 lg:grid-cols-2">
          <Card title="Temperature (°C)" subtitle="measured · per-discharge mean and peak"><L keys={[{ key: 'dis_t_mean', name: 'mean', color: '#f59e0b' }, { key: 'dis_t_max', name: 'peak', color: '#ef4444' }]} /></Card>
          <Card title="Current (A)" subtitle="measured · discharge mean"><L keys={[{ key: 'dis_i_mean', name: 'mean current', color: '#06b6d4' }]} /></Card>
          <Card title="Voltage (V)" subtitle="measured · discharge"><L keys={[{ key: 'dis_v_mean', name: 'mean', color: '#a855f7' }, { key: 'dis_v_min', name: 'min', color: '#64748b' }]} /></Card>
          <Card title="Discharge duration (s)" subtitle="measured"><L keys={[{ key: 'dis_duration_s', name: 'duration', color: '#14b8a6' }]} unit="" tipFmt={0} /></Card>
        </div>)}
        {tab === 'res' && (<div className="grid gap-4 lg:grid-cols-2">
          <Card title="Electrolyte resistance Re (Ω)" subtitle="from impedance measurements, forward-filled"><L keys={[{ key: 're_ohm', name: 'Re', color: '#f59e0b' }]} tipFmt={4} /></Card>
          <Card title="Charge-transfer resistance Rct (Ω)" subtitle="from impedance measurements, forward-filled"><L keys={[{ key: 'rct_ohm', name: 'Rct', color: '#ef4444' }]} tipFmt={4} /></Card>
        </div>)}
        {tab === 'fc' && (
          <Card title={`Predicted vs actual SOH — ${horizon}-cycle horizon`} subtitle="Out-of-fold: this battery was held out of the models that produced these predictions">
            <Async state={perf}>{() => pts.length === 0 ? <Empty title="No held-out predictions for this battery" /> : (
              <Chart height={300}><LineChart data={pts}><CartesianGrid {...gridProps} /><XAxis dataKey="target_cycle" {...axisProps} type="number" domain={['dataMin', 'dataMax']} /><YAxis domain={['auto', 'auto']} {...axisProps} unit="%" />
                <Tooltip {...tipProps} formatter={(v) => fmt(v, 2) + ' %'} labelFormatter={(l) => `target cycle ${l}`} /><Legend wrapperStyle={{ fontSize: 11 }} />
                <Line dataKey="y_true" name="Actual" stroke="#14b8a6" dot={false} strokeWidth={2} /><Line dataKey="y_pred" name={`Predicted (${perf.data.best_model})`} stroke="#f59e0b" dot={false} strokeWidth={1.6} strokeDasharray="5 3" /></LineChart></Chart>)}</Async>
          </Card>)}
      </>)}</Async>
    </div>
  )
}
