import { useEffect, useState } from 'react'
import {
  Activity,
  AlertCircle,
  BatteryCharging,
  BatteryMedium,
  CheckCircle2,
  Clock,
  Compass,
  Flame,
  Gauge,
  Info,
  Lightbulb,
  MapPin,
  RefreshCw,
  Route,
  Shield,
  Sparkles,
  Thermometer,
  Zap,
} from 'lucide-react'
import { Area, AreaChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Badge, Button, Card, Field, Loading, PageTitle } from '../components/ui.jsx'
import { api } from '../services/api.js'

const CLIMATE_OPTIONS = [
  { value: 10, label: 'Cold (< 15°C / 59°F)' },
  { value: 24, label: 'Moderate / Temperate (20 - 25°C / 77°F)' },
  { value: 34, label: 'Warm / Tropical (30 - 35°C / 95°F)' },
  { value: 40, label: 'Hot / Desert (> 38°C / 100°F)' },
]

const CHARGE_TARGET_OPTIONS = [
  { value: 80, label: '80% (Recommended for longevity)' },
  { value: 90, label: '90% (Balanced)' },
  { value: 100, label: '100% (Daily full charge)' },
]

export default function EvCalculator() {
  const [presets, setPresets] = useState([])
  const [selectedPresetId, setSelectedPresetId] = useState('tesla_model_3')

  // Form State
  const [capacityKwh, setCapacityKwh] = useState(60.0)
  const [odometerKm, setOdometerKm] = useState(45000)
  const [ageYears, setAgeYears] = useState(3.0)
  const [fastChargePct, setFastChargePct] = useState(20)
  const [ambientTempC, setAmbientTempC] = useState(24)
  const [chargeLimitPct, setChargeLimitPct] = useState(90)
  const [efficiencyWhKm, setEfficiencyWhKm] = useState(145)
  const [ratedRangeKm, setRatedRangeKm] = useState(491)

  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  // Fetch Presets on Mount
  useEffect(() => {
    api.evPresets()
      .then((res) => {
        if (res?.presets?.length > 0) {
          setPresets(res.presets)
          applyPreset(res.presets[0])
        }
      })
      .catch((err) => console.error('Failed to load presets:', err))
  }, [])

  const applyPreset = (p) => {
    setSelectedPresetId(p.id)
    setCapacityKwh(p.capacity_kwh)
    setEfficiencyWhKm(p.efficiency_wh_km)
    setRatedRangeKm(p.rated_range_km)
  }

  const handleCalculate = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.evPredict({
        capacity_kwh: Number(capacityKwh),
        odometer_km: Number(odometerKm),
        age_years: Number(ageYears),
        fast_charge_pct: Number(fastChargePct),
        ambient_temp_c: Number(ambientTempC),
        charge_limit_pct: Number(chargeLimitPct),
        efficiency_wh_km: Number(efficiencyWhKm),
        rated_range_km: Number(ratedRangeKm),
      })
      setResult(data)
    } catch (e) {
      setError(e?.message || 'Failed to calculate EV battery health.')
    } finally {
      setLoading(false)
    }
  }

  // Calculate automatically once initial data is available
  useEffect(() => {
    handleCalculate()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const breakdownData = result?.breakdown ? [
    { name: 'Calendar Aging', value: result.breakdown.calendar_loss_pct, color: '#38bdf8' },
    { name: 'Cycle Usage Wear', value: result.breakdown.cycle_loss_pct, color: '#a855f7' },
    { name: 'Fast Charge Thermal Stress', value: result.breakdown.fast_charge_loss_pct, color: '#f59e0b' },
    { name: 'Climate Extremes', value: result.breakdown.thermal_loss_pct, color: '#ef4444' },
  ].filter((d) => d.value > 0) : []

  return (
    <div className="mx-auto max-w-7xl space-y-6">
      <PageTitle
        title="Real-World EV SOH & Range Calculator"
        subtitle="Predict your vehicle's State of Health (SOH), remaining battery capacity, real-world range, and expected lifespan to 80% EOL using your driving and charging habits."
      />

      {/* Preset vehicle selector */}
      <Card title="Quick Vehicle Presets" subtitle="Select your EV model or customize pack parameters manually below:">
        <div className="flex flex-wrap gap-2 pt-1">
          {presets.map((p) => (
            <button
              key={p.id}
              onClick={() => applyPreset(p)}
              className={`flex items-center gap-2 rounded-lg border px-3 py-2 text-xs font-medium transition-all ${
                selectedPresetId === p.id
                  ? 'border-brand-500 bg-brand-50/70 text-brand-700 shadow-sm dark:border-brand-500 dark:bg-brand-950/40 dark:text-brand-300'
                  : 'border-slate-200 bg-white text-slate-700 hover:border-slate-300 hover:bg-slate-50 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-300 dark:hover:bg-slate-800'
              }`}
            >
              <Zap className={`h-3.5 w-3.5 ${selectedPresetId === p.id ? 'text-brand-600 dark:text-brand-400' : 'text-slate-400'}`} />
              <span>{p.name}</span>
              <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[10px] text-slate-500 dark:bg-slate-800 dark:text-slate-400">
                {p.capacity_kwh} kWh
              </span>
            </button>
          ))}
          <button
            onClick={() => setSelectedPresetId('custom')}
            className={`flex items-center gap-2 rounded-lg border px-3 py-2 text-xs font-medium transition-all ${
              selectedPresetId === 'custom'
                ? 'border-brand-500 bg-brand-50/70 text-brand-700 dark:border-brand-500 dark:bg-brand-950/40 dark:text-brand-300'
                : 'border-slate-200 bg-white text-slate-700 hover:border-slate-300 hover:bg-slate-50 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-300'
            }`}
          >
            <span>Custom EV / Pack</span>
          </button>
        </div>
      </Card>

      {/* Main Grid: Input Form + Real-time Results */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        {/* Left Column: Form Controls */}
        <div className="space-y-4 lg:col-span-5">
          <Card title="Vehicle & Usage Parameters">
            <div className="space-y-4">
              {/* Odometer */}
              <div>
                <div className="flex items-center justify-between text-xs font-medium text-slate-700 dark:text-slate-300">
                  <span className="flex items-center gap-1.5"><Route className="h-3.5 w-3.5 text-slate-400" /> Total Odometer Distance</span>
                  <span className="font-semibold text-brand-600 dark:text-brand-400">{Number(odometerKm).toLocaleString()} km</span>
                </div>
                <input
                  type="range"
                  min="500"
                  max="250000"
                  step="1000"
                  value={odometerKm}
                  onChange={(e) => setOdometerKm(Number(e.target.value))}
                  className="mt-2 w-full accent-brand-600"
                />
                <div className="flex justify-between text-[10px] text-slate-400">
                  <span>500 km</span>
                  <span>100,000 km</span>
                  <span>250,000 km</span>
                </div>
              </div>

              {/* Age */}
              <div>
                <div className="flex items-center justify-between text-xs font-medium text-slate-700 dark:text-slate-300">
                  <span className="flex items-center gap-1.5"><Clock className="h-3.5 w-3.5 text-slate-400" /> Vehicle Age</span>
                  <span className="font-semibold text-brand-600 dark:text-brand-400">{ageYears} Years ({Math.round(ageYears * 12)} months)</span>
                </div>
                <input
                  type="range"
                  min="0.2"
                  max="12"
                  step="0.2"
                  value={ageYears}
                  onChange={(e) => setAgeYears(Number(e.target.value))}
                  className="mt-2 w-full accent-brand-600"
                />
                <div className="flex justify-between text-[10px] text-slate-400">
                  <span>3 mo</span>
                  <span>5 yrs</span>
                  <span>12 yrs</span>
                </div>
              </div>

              {/* Fast Charging % */}
              <div>
                <div className="flex items-center justify-between text-xs font-medium text-slate-700 dark:text-slate-300">
                  <span className="flex items-center gap-1.5"><Zap className="h-3.5 w-3.5 text-amber-500" /> Fast Charging (DCFC) Ratio</span>
                  <span className="font-semibold text-amber-600 dark:text-amber-400">{fastChargePct}% DC</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="5"
                  value={fastChargePct}
                  onChange={(e) => setFastChargePct(Number(e.target.value))}
                  className="mt-2 w-full accent-amber-500"
                />
                <div className="flex justify-between text-[10px] text-slate-400">
                  <span>0% (Home AC only)</span>
                  <span>50%</span>
                  <span>100% (Highways)</span>
                </div>
              </div>

              {/* Charge Target Limit */}
              <Field label="Daily Charging Target" hint="Restricting regular charging to 80% slows calendar degradation significantly.">
                <select
                  value={chargeLimitPct}
                  onChange={(e) => setChargeLimitPct(Number(e.target.value))}
                  className="w-full rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-sm text-slate-900 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
                >
                  {CHARGE_TARGET_OPTIONS.map((o) => (
                    <option key={o.value} value={o.value}>{o.label}</option>
                  ))}
                </select>
              </Field>

              {/* Climate */}
              <Field label="Operating Climate / Temperature">
                <select
                  value={ambientTempC}
                  onChange={(e) => setAmbientTempC(Number(e.target.value))}
                  className="w-full rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-sm text-slate-900 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
                >
                  {CLIMATE_OPTIONS.map((o) => (
                    <option key={o.value} value={o.value}>{o.label}</option>
                  ))}
                </select>
              </Field>

              {/* Original Pack Size & Rated Range */}
              <div className="grid grid-cols-2 gap-3 pt-1">
                <div>
                  <label className="block text-xs font-medium text-slate-600 dark:text-slate-300">Pack Capacity (kWh)</label>
                  <input
                    type="number"
                    step="0.5"
                    value={capacityKwh}
                    onChange={(e) => { setCapacityKwh(Number(e.target.value)); setSelectedPresetId('custom') }}
                    className="mt-1 w-full rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-sm text-slate-900 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-600 dark:text-slate-300">Original Rated Range (km)</label>
                  <input
                    type="number"
                    step="5"
                    value={ratedRangeKm}
                    onChange={(e) => { setRatedRangeKm(Number(e.target.value)); setSelectedPresetId('custom') }}
                    className="mt-1 w-full rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-sm text-slate-900 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
                  />
                </div>
              </div>

              <div className="pt-2">
                <Button onClick={handleCalculate} disabled={loading} className="w-full justify-center py-2">
                  {loading ? <Loading label="Calculating..." /> : <><Sparkles className="h-4 w-4" /> Calculate Battery SOH &amp; Lifespan</>}
                </Button>
              </div>
            </div>
          </Card>
        </div>

        {/* Right Column: Prediction Metrics & Charts */}
        <div className="space-y-6 lg:col-span-7">
          {error && (
            <div className="rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800 dark:border-rose-900 dark:bg-rose-950 dark:text-rose-200">
              <AlertCircle className="mb-1 inline h-4 w-4 mr-1.5" />
              {error}
            </div>
          )}

          {result && (
            <>
              {/* Top Hero KPI Cards */}
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                {/* SOH */}
                <div className="rounded-xl border border-brand-200 bg-gradient-to-br from-brand-50 to-white p-4 shadow-sm dark:border-brand-900/50 dark:from-slate-900 dark:to-slate-900/80">
                  <div className="flex items-center justify-between text-xs font-medium text-slate-500 dark:text-slate-400">
                    <span>Predicted SOH</span>
                    <Activity className="h-4 w-4 text-brand-600 dark:text-brand-400" />
                  </div>
                  <div className="mt-2 text-2xl font-bold text-slate-900 dark:text-white">
                    {result.soh_predicted}%
                  </div>
                  <div className="mt-1 text-[11px] text-slate-500">
                    90% CI: [{result.soh_lower_bound}% – {result.soh_upper_bound}%]
                  </div>
                </div>

                {/* Usable Capacity */}
                <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
                  <div className="flex items-center justify-between text-xs font-medium text-slate-500 dark:text-slate-400">
                    <span>Usable Capacity</span>
                    <BatteryMedium className="h-4 w-4 text-emerald-500" />
                  </div>
                  <div className="mt-2 text-2xl font-bold text-slate-900 dark:text-white">
                    {result.current_usable_kwh} <span className="text-sm font-normal text-slate-500">kWh</span>
                  </div>
                  <div className="mt-1 text-[11px] text-rose-500 dark:text-rose-400">
                    -{result.capacity_loss_kwh} kWh lost
                  </div>
                </div>

                {/* Remaining Range */}
                <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
                  <div className="flex items-center justify-between text-xs font-medium text-slate-500 dark:text-slate-400">
                    <span>Est. Range Now</span>
                    <Compass className="h-4 w-4 text-sky-500" />
                  </div>
                  <div className="mt-2 text-2xl font-bold text-slate-900 dark:text-white">
                    {result.current_estimated_range_km} <span className="text-sm font-normal text-slate-500">km</span>
                  </div>
                  <div className="mt-1 text-[11px] text-slate-500">
                    Original: {result.rated_range_km} km
                  </div>
                </div>

                {/* Lifespan to 80% EOL */}
                <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
                  <div className="flex items-center justify-between text-xs font-medium text-slate-500 dark:text-slate-400">
                    <span>Lifespan to 80%</span>
                    <Shield className="h-4 w-4 text-violet-500" />
                  </div>
                  <div className="mt-2 text-2xl font-bold text-slate-900 dark:text-white">
                    ~{Math.round(result.lifespan_estimate.km_to_eol_80).toLocaleString()} <span className="text-sm font-normal text-slate-500">km</span>
                  </div>
                  <div className="mt-1 text-[11px] text-emerald-600 dark:text-emerald-400 font-medium">
                    +{result.lifespan_estimate.years_to_eol_80} years left
                  </div>
                </div>
              </div>

              {/* Degradation Trajectory Chart */}
              <Card
                title="Projected Battery SOH Trajectory (0 to 200,000+ km)"
                subtitle="Future SOH and real-world range estimation based on current driving & charging patterns."
              >
                <div className="h-64 w-full pt-2">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={result.trajectory} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
                      <defs>
                        <linearGradient id="sohGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0.0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                      <XAxis
                        dataKey="odometer_km"
                        tickFormatter={(v) => `${Math.round(v / 1000)}k`}
                        label={{ value: 'Odometer (km)', position: 'insideBottomRight', offset: -5, fontSize: 11 }}
                      />
                      <YAxis domain={[60, 100]} label={{ value: 'SOH %', angle: -90, position: 'insideLeft', offset: 15, fontSize: 11 }} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                        formatter={(val, name) => [
                          name === 'soh' ? `${val}%` : name === 'range_km' ? `${val} km` : `${val} kWh`,
                          name === 'soh' ? 'State of Health' : name === 'range_km' ? 'Estimated Range' : 'Usable Pack',
                        ]}
                        labelFormatter={(v) => `Odometer: ${Number(v).toLocaleString()} km`}
                      />
                      <Area type="monotone" dataKey="soh" stroke="#0ea5e9" strokeWidth={2.5} fillOpacity={1} fill="url(#sohGradient)" name="soh" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-3 flex items-center justify-between border-t border-slate-100 pt-3 text-xs text-slate-500 dark:border-slate-800">
                  <span className="flex items-center gap-1.5"><CheckCircle2 className="h-4 w-4 text-emerald-500" /> Battery Condition: <b className="text-slate-800 dark:text-slate-200">{result.status}</b></span>
                  <span>Equivalent Full Cycles: <b>{result.equivalent_full_cycles} EFC</b></span>
                </div>
              </Card>

              {/* Degradation Breakdown & AI Recommendations */}
              <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
                {/* Degradation Breakdown */}
                <Card title="Degradation Factor Breakdown" subtitle="Where your capacity loss is originating:">
                  <div className="h-44 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={breakdownData}
                          dataKey="value"
                          nameKey="name"
                          cx="50%"
                          cy="50%"
                          innerRadius={38}
                          outerRadius={65}
                          paddingAngle={3}
                        >
                          {breakdownData.map((entry, idx) => (
                            <Cell key={`cell-${idx}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip
                          formatter={(val) => [`${val}% loss`, 'Degradation']}
                          contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="space-y-1.5 pt-1 text-xs">
                    {breakdownData.map((d) => (
                      <div key={d.name} className="flex items-center justify-between text-slate-600 dark:text-slate-400">
                        <span className="flex items-center gap-2">
                          <span className="h-2 w-2 rounded-full" style={{ backgroundColor: d.color }} />
                          {d.name}
                        </span>
                        <span className="font-semibold text-slate-900 dark:text-white">-{d.value}%</span>
                      </div>
                    ))}
                  </div>
                </Card>

                {/* AI Battery Care Tips */}
                <Card title="AI Battery Care Insights" subtitle="Tailored to your usage pattern:">
                  <div className="space-y-3 pt-1">
                    {result.recommendations?.map((rec, i) => (
                      <div key={i} className="rounded-lg border border-slate-200 bg-slate-50/70 p-3 text-xs dark:border-slate-800 dark:bg-slate-950/60">
                        <div className="flex items-center justify-between">
                          <span className="flex items-center gap-1.5 font-semibold text-slate-900 dark:text-white">
                            <Lightbulb className="h-3.5 w-3.5 text-amber-500" />
                            {rec.title}
                          </span>
                          <Badge tone={rec.impact === 'High' ? 'bad' : rec.impact === 'Medium' ? 'warn' : 'good'}>
                            {rec.impact} Impact
                          </Badge>
                        </div>
                        <p className="mt-1.5 leading-relaxed text-slate-600 dark:text-slate-400">
                          {rec.desc}
                        </p>
                      </div>
                    ))}
                  </div>
                </Card>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
