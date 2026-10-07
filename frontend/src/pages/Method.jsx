import { Async, Card, DemoBanner, PageTitle, useAsync } from '../components/ui.jsx'
import { useApp } from '../context/AppContext.jsx'
import { api } from '../services/api.js'

const TERMS = [
  ['SOH — State of Health', 'Current capacity ÷ reference (rated) capacity × 100. 100 % = like new.'],
  ['EOL — End of life', 'The SOH threshold that defines a “dead” battery. Default 80 %, configurable.'],
  ['RUL — Remaining Useful Life', 'Estimated cycles left until SOH first reaches the EOL threshold. An ML estimate with real uncertainty, not a physical measurement.'],
  ['Degradation rate', 'Percentage points of SOH lost per cycle; rolling means over 10/25/50 cycles reduce noise.'],
  ['Leave-one-battery-out', 'Each battery is held out entirely and scored by models trained on the others — the leakage-safe way to test on a battery the model has never seen.'],
  ['pp', 'Percentage points of SOH (an error of 0.5 pp means 0.5 SOH-percent).'],
]
const STEPS = ['Dataset adapter → common cycle table', 'Cleaning with an explicit quality report', 'SOH, capacity loss, degradation rate, RUL targets', 'Trailing-window historical features (no look-ahead)',
  'Drop exact-duplicate features', 'Leave-one-battery-out evaluation with battery-grouped tuning', 'Linear / Random Forest / XGBoost + no-model references', 'Best model refit on all batteries → saved artifact',
  'SHAP explainability + residual band', 'FastAPI serves saved artifacts → this dashboard']
const LIMITS = ['Laboratory cells are not real EVs: no drive cycles, thermal management, calendar ageing or mixed chemistries.', 'Few batteries ⇒ results vary a lot across held-out cells.',
  'Model selection uses the same held-out scores that are reported, so they are mildly optimistic.', 'Residual bands are coarse; RUL has no interval yet.', 'SHAP describes the model — it is not evidence of physical causation.']

export default function Method() {
  const { dataset, dsId } = useApp()
  const m = useAsync(() => api.models(dsId), [dsId])
  return (
    <div className="mx-auto max-w-6xl">
      <PageTitle title="Method & Data" subtitle="How the numbers in this dashboard are produced" />
      <DemoBanner dataset={dataset} />
      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Glossary"><dl className="space-y-2.5 text-xs">{TERMS.map(([t, d]) => <div key={t}><dt className="font-semibold text-slate-800 dark:text-slate-100">{t}</dt><dd className="text-slate-500 dark:text-slate-400">{d}</dd></div>)}</dl></Card>
        <Card title="Pipeline"><ol className="space-y-1.5 text-xs">{STEPS.map((s, i) => <li key={s} className="flex gap-2"><span className="num grid h-5 w-5 shrink-0 place-items-center rounded-full bg-brand-100 text-[10px] font-semibold text-brand-700 dark:bg-brand-700/30 dark:text-brand-400">{i + 1}</span><span className="text-slate-600 dark:text-slate-300">{s}</span></li>)}</ol></Card>
        <Async state={m}>{(d) => (<>
          <Card title="Validation (temporal-leakage defence)"><p className="text-xs leading-relaxed text-slate-600 dark:text-slate-300">{d.validation}</p>
            <p className="mt-2 text-xs text-slate-500">{d.model_selection_note}</p>
            <p className="mt-2 text-xs text-slate-500">Data: {d.batteries.length} batteries · {d.n_cycles} cycle records · Python {d.versions.python} · scikit-learn {d.versions.sklearn} · numpy {d.versions.numpy} · pandas {d.versions.pandas}</p></Card>
          <Card title="Trained models"><table className="w-full text-left text-xs"><thead className="text-slate-500"><tr><th className="py-1">Task</th><th>Model</th><th>Features</th></tr></thead>
            <tbody>{d.models.map((x) => <tr key={x.id} className="border-t border-slate-100 dark:border-slate-800"><td className="py-1.5">{x.task}{x.horizon ? ` · ${x.horizon} cyc` : ''}</td><td className="font-medium">{x.model}</td><td className="num">{x.n_features}</td></tr>)}</tbody></table></Card>
          <Card className="lg:col-span-2" title={`Features used (${Object.keys(d.feature_docs).length})`} subtitle="Every feature is trailing-window: it only contains information available at the prediction cycle">
            <div className="grid gap-x-6 gap-y-1.5 text-xs sm:grid-cols-2">{Object.entries(d.feature_docs).map(([k, v]) => <div key={k}><span className="font-mono text-slate-800 dark:text-slate-100">{k}</span> <span className="text-slate-500">— {v}</span></div>)}</div></Card>
          <Card title="Considered but not used"><ul className="space-y-1.5 text-xs text-slate-500">{Object.entries(d.excluded_features).map(([k, v]) => <li key={k}><span className="font-mono text-slate-700 dark:text-slate-200">{k}</span> — {v}</li>)}
            {Object.entries(d.collinear_dropped || {}).map(([k, v]) => <li key={k}><span className="font-mono text-slate-700 dark:text-slate-200">{k}</span> — exact duplicate of {v} (auto-removed)</li>)}</ul></Card>
        </>)}</Async>
        <Card title="Limitations"><ul className="list-disc space-y-1.5 pl-4 text-xs text-slate-600 dark:text-slate-300">{LIMITS.map((l) => <li key={l}>{l}</li>)}</ul></Card>
      </div>
    </div>
  )
}
