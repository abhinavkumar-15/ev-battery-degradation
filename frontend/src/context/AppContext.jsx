import { createContext, useContext, useEffect, useMemo, useState } from 'react'
import { api } from '../services/api.js'

const Ctx = createContext(null)
export const useApp = () => useContext(Ctx)

export function AppProvider({ children }) {
  const [theme, setTheme] = useState(() => localStorage.getItem('vg-theme') || 'dark')
  const [datasets, setDatasets] = useState(null)
  const [dsId, setDsId] = useState(null)
  const [batteries, setBatteries] = useState([])
  const [battery, setBattery] = useState(null)
  const [error, setError] = useState(null)

  const [refreshKey, setRefreshKey] = useState(0)
  const reload = () => { setError(null); setRefreshKey((k) => k + 1) }

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark')
    localStorage.setItem('vg-theme', theme)
  }, [theme])

  useEffect(() => {
    api.datasets()
      .then((r) => {
        setError(null)
        setDatasets(r.datasets)
        setDsId((curr) => curr && r.datasets.some((d) => d.id === curr) ? curr : (r.datasets[0]?.id ?? null))
      })
      .catch((e) => setError(e))
  }, [refreshKey])

  useEffect(() => {
    if (!dsId) return
    setBatteries([])
    setBattery(null)
    api.batteries(dsId)
      .then((r) => {
        setError(null)
        setBatteries(r.batteries)
        setBattery(r.batteries[0]?.id ?? null)
      })
      .catch((e) => setError(e))
  }, [dsId, refreshKey])

  const dataset = useMemo(() => datasets?.find((d) => d.id === dsId) ?? null, [datasets, dsId])
  const value = { theme, setTheme, datasets, dataset, dsId, setDsId, batteries, battery, setBattery, error, reload }
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}
