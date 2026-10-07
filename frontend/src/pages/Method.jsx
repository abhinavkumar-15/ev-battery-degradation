import { Async, Card, DemoBanner, PageTitle, useAsync } from '../components/ui.jsx'
import { useApp } from '../context/AppContext.jsx'
import { api } from '../services/api.js'

const TERMS = [
  ['SOH — State of Health', 'Current capacity ÷ reference (rated) capacity × 100. 100 % = like new.'],
  ['EOL — End of life', 'The SOH threshold that defines retirement for EV traction packs. Default 80 %, configurable.'],
  ['RUL — Remaining Useful Life', 'Estimated cycles or kilometers left until SOH first reaches the EOL threshold.'],
  ['EFC — Equivalent Full Cycles', 'Total kWh cycled through pack ÷ rated pack capacity = (Odometer km × Wh/km) ÷ (1000 × Capacity kWh).'],
  ['DOD — Depth of Discharge', 'The percentage of battery capacity discharged per cycle. Frequent shallow top-ups minimize DOD and crystal stress.'],
  ['Leave-one-battery-out', 'Each battery is held out entirely and scored by models trained on the others — leakage-safe validation.'],
]

const PIPELINES = [
  {
    title: 'Laboratory Cycler Pipeline (NASA / Synthetic Cycler Data)',
    steps: [
      'Raw cell cycler telemetry (.mat / CSV) → standardized cycle records',
      'Automated quality validation & out-of-bounds cleaning',
      'Trailing-window feature engineering (voltage, current, temperature, resistance slopes)',
      'Leave-one-battery-out cross validation across Linear, Random Forest, XGBoost & HistGBDT',
      'Model refitting with SHAP explainability & residual uncertainty bands',
    ],
  },
  {
    title: 'Real-World EV SOH Calculator Engine',
    steps: [
      'Physics-informed degradation synthesis incorporating Arrhenius calendar aging + EFC cycle wear',
      'Depth of Discharge (DOD) modeling derived from user Charging Frequency patterns',
      'Thermal & mechanical C-rate degradation penalties from DC Fast Charging (DCFC) %',
      'Gradient Boosting Ensemble with 10th & 90th Quantile Regressors for confidence intervals',
      'Trajectory extrapolation up to 250,000+ km with personalized AI battery longevity tips',
    ],
  },
]

const LIMITS = [
  'Laboratory battery cycler tests operate under controlled cell fixtures; real-world vehicle packs experience complex vibration, BMS balancing, and active liquid thermal management.',
  'Real-world degradation rate varies by cell chemistry (LFP flat voltage curve vs NMC high energy density) and thermal cooling architecture.',
  'Quantile uncertainty bands offer empirical bounds based on average fleet wear distributions.',
]

export default function Method() {
  const { dataset, dsId } = useApp()
  const m = useAsync(() => api.models(dsId), [dsId])

  return (
    <div className="mx-auto max-w-6xl space-y-4">
      <PageTitle title="Method & Architecture" subtitle="Physics, Machine Learning pipelines, and validation methodologies behind VoltGuard AI" />
      <DemoBanner dataset={dataset} />

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Key Battery Degradation Terms">
          <dl className="space-y-2.5 text-xs">
            {TERMS.map(([t, d]) => (
              <div key={t}>
                <dt className="font-semibold text-slate-800 dark:text-slate-100">{t}</dt>
                <dd className="text-slate-500 dark:text-slate-400">{d}</dd>
              </div>
            ))}
          </dl>
        </Card>

        <Card title="System Pipelines">
          <div className="space-y-4 text-xs">
            {PIPELINES.map((p, idx) => (
              <div key={idx} className="rounded-lg border border-slate-100 bg-slate-50/60 p-3 dark:border-slate-800 dark:bg-slate-950/40">
                <h4 className="font-semibold text-slate-900 dark:text-white mb-2">{p.title}</h4>
                <ol className="space-y-1.5 pl-1">
                  {p.steps.map((s, i) => (
                    <li key={s} className="flex gap-2 text-slate-600 dark:text-slate-300">
                      <span className="grid h-4 w-4 shrink-0 place-items-center rounded-full bg-brand-100 text-[9px] font-semibold text-brand-700 dark:bg-brand-700/30 dark:text-brand-400">
                        {i + 1}
                      </span>
                      <span>{s}</span>
                    </li>
                  ))}
                </ol>
              </div>
            ))}
          </div>
        </Card>

        <Async state={m}>{(d) => (<>
          <Card title="Validation (Temporal-Leakage Defense)">
            <p className="text-xs leading-relaxed text-slate-600 dark:text-slate-300">{d.validation}</p>
            <p className="mt-2 text-xs text-slate-500">{d.model_selection_note}</p>
            <p className="mt-2 text-xs text-slate-500">Data: {d.batteries.length} batteries · {d.n_cycles} cycle records · Python {d.versions.python} · scikit-learn {d.versions.sklearn}</p>
          </Card>

          <Card title="Trained Laboratory Models">
            <table className="w-full text-left text-xs">
              <thead className="text-slate-500">
                <tr><th className="py-1">Task</th><th>Model</th><th>Features</th></tr>
              </thead>
              <tbody>
                {d.models.map((x) => (
                  <tr key={x.id} className="border-t border-slate-100 dark:border-slate-800">
                    <td className="py-1.5">{x.task}{x.horizon ? ` · ${x.horizon} cyc` : ''}</td>
                    <td className="font-medium">{x.model}</td>
                    <td className="num">{x.n_features}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card className="lg:col-span-2" title={`Laboratory Features (${Object.keys(d.feature_docs).length})`} subtitle="Every cycler feature is trailing-window: strictly zero look-ahead">
            <div className="grid gap-x-6 gap-y-1.5 text-xs sm:grid-cols-2">
              {Object.entries(d.feature_docs).map(([k, v]) => (
                <div key={k}>
                  <span className="font-mono text-slate-800 dark:text-slate-100">{k}</span> <span className="text-slate-500">— {v}</span>
                </div>
              ))}
            </div>
          </Card>
        </>)}</Async>

        <Card title="Methodological Limitations & Caveats" className="lg:col-span-2">
          <ul className="list-disc space-y-1.5 pl-4 text-xs text-slate-600 dark:text-slate-300">
            {LIMITS.map((l, i) => <li key={i}>{l}</li>)}
          </ul>
        </Card>
      </div>
    </div>
  )
}
