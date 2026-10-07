import { ResponsiveContainer } from 'recharts'

export const COLORS = ['#14b8a6', '#6366f1', '#f59e0b', '#ef4444', '#06b6d4', '#a855f7', '#84cc16', '#ec4899']
export const axisProps = { stroke: 'var(--axis)', tick: { fill: 'var(--axis)', fontSize: 11 }, tickLine: false }
export const gridProps = { stroke: 'var(--grid)', strokeDasharray: '3 3' }
export const tipProps = {
  contentStyle: { background: 'var(--tip-bg)', border: '1px solid var(--tip-bd)', borderRadius: 8, fontSize: 12, color: 'var(--tip-fg)' },
  labelStyle: { color: 'var(--tip-fg)' },
}
export const Chart = ({ height = 260, children }) => (
  <div style={{ height }} className="w-full"><ResponsiveContainer width="100%" height="100%">{children}</ResponsiveContainer></div>
)
