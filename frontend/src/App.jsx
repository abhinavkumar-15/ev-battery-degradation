import { useState } from 'react'
import { Activity, BarChart3, BookOpen, Brain, Database, Gauge, LineChart, Menu, Moon, Sun, X, Zap } from 'lucide-react'
import { useApp } from './context/AppContext.jsx'
import { Badge, ErrorBox, Loading, Select } from './components/ui.jsx'
import Overview from './pages/Overview.jsx'
import BatteryAnalysis from './pages/BatteryAnalysis.jsx'
import Prediction from './pages/Prediction.jsx'
import Explainability from './pages/Explainability.jsx'
import DatasetExplorer from './pages/DatasetExplorer.jsx'
import ModelPerformance from './pages/ModelPerformance.jsx'
import Method from './pages/Method.jsx'

const NAV = [
  { id: 'overview', label: 'Overview', icon: Gauge, C: Overview },
  { id: 'analysis', label: 'Battery Analysis', icon: LineChart, C: BatteryAnalysis },
  { id: 'prediction', label: 'Prediction', icon: Activity, C: Prediction },
  { id: 'explain', label: 'Explainability', icon: Brain, C: Explainability },
  { id: 'dataset', label: 'Dataset Explorer', icon: Database, C: DatasetExplorer },
  { id: 'performance', label: 'Model Performance', icon: BarChart3, C: ModelPerformance },
  { id: 'method', label: 'Method & Data', icon: BookOpen, C: Method },
]

export default function App() {
  const { theme, setTheme, datasets, dataset, dsId, setDsId, batteries, battery, setBattery, error } = useApp()
  const [page, setPage] = useState(() => location.hash.slice(1) || 'overview')
  const [open, setOpen] = useState(false)
  const go = (id) => { setPage(id); location.hash = id; setOpen(false) }
  const Page = (NAV.find((n) => n.id === page) || NAV[0]).C

  const nav = (
    <nav className="flex flex-col gap-1 p-3" aria-label="Main">
      {NAV.map(({ id, label, icon: Icon }) => (
        <button key={id} onClick={() => go(id)} aria-current={page === id ? 'page' : undefined}
          className={`flex items-center gap-2.5 rounded-md px-3 py-2 text-left text-sm font-medium transition-colors ${
            page === id ? 'bg-brand-50 text-brand-700 dark:bg-brand-700/20 dark:text-brand-400'
              : 'text-slate-600 hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800'}`}>
          <Icon className="h-4 w-4" aria-hidden /> {label}
        </button>
      ))}
    </nav>
  )

  return (
    <div className="flex h-full">
      <aside className="hidden w-60 shrink-0 flex-col border-r border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900 lg:flex">
        <div className="flex items-center gap-2 px-5 py-4">
          <span className="grid h-8 w-8 place-items-center rounded-lg bg-brand-600 text-white"><Zap className="h-4 w-4" /></span>
          <div><div className="text-sm font-semibold text-slate-900 dark:text-white">VoltGuard AI</div>
            <div className="text-[11px] text-slate-500">Battery SOH &amp; RUL</div></div>
        </div>
        {nav}
        <p className="mt-auto px-5 pb-4 text-[11px] leading-snug text-slate-500">Estimates from a statistical model — not guarantees. Lab data ≠ real-world EV behaviour.</p>
      </aside>

      {open && (
        <div className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal="true">
          <div className="absolute inset-0 bg-black/50" onClick={() => setOpen(false)} />
          <aside className="absolute left-0 top-0 h-full w-64 bg-white dark:bg-slate-900">
            <div className="flex items-center justify-between px-5 py-4"><b className="text-sm">VoltGuard AI</b>
              <button onClick={() => setOpen(false)} aria-label="Close menu"><X className="h-4 w-4" /></button></div>
            {nav}
          </aside>
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex flex-wrap items-center gap-3 border-b border-slate-200 bg-white px-4 py-2.5 dark:border-slate-800 dark:bg-slate-900">
          <button className="lg:hidden" onClick={() => setOpen(true)} aria-label="Open menu"><Menu className="h-5 w-5" /></button>
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500">Dataset</span>
            <div className="w-40">{datasets && <Select aria-label="Dataset" value={dsId} onChange={setDsId}
              options={datasets.map((d) => ({ value: d.id, label: d.id + (d.is_demo ? ' (demo)' : '') }))} />}</div>
            <span className="ml-2 text-xs text-slate-500">Battery</span>
            <div className="w-28">{batteries.length > 0 && <Select aria-label="Battery" value={battery} onChange={setBattery} options={batteries.map((b) => b.id)} />}</div>
          </div>
          <div className="ml-auto flex items-center gap-2">
            {dataset && <Badge tone={dataset.is_demo ? 'warn' : 'good'}>{dataset.is_demo ? 'Demo / Sample Data' : 'Real dataset'}</Badge>}
            <button onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} aria-label="Toggle theme"
              className="rounded-md p-2 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800">
              {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}</button>
          </div>
        </header>

        <main className="min-h-0 flex-1 overflow-y-auto p-4 lg:p-6">
          {error ? (
            <div className="mx-auto max-w-xl pt-10"><ErrorBox error={error} />
              <p className="mt-3 text-xs text-slate-500">Start the API with <code>uvicorn backend.app.main:app --port 8000</code> from the repo root and make sure a model has been trained (<code>python -m src.pipeline --dataset demo</code>).</p></div>
          ) : !datasets ? <Loading label="Connecting to API…" /> : datasets.length === 0 ? (
            <div className="mx-auto max-w-xl pt-10 text-sm">No trained models found. Run the training pipeline first (see README).</div>
          ) : !dataset || !battery ? <Loading /> : <Page key={`${dsId}-${page}`} />}
        </main>
      </div>
    </div>
  )
}
