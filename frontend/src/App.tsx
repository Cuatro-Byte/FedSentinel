import { useEffect, useState } from 'react'
import {
  Activity,
  ClipboardList,
  Database,
  Menu,
  Moon,
  PanelLeftClose,
  RotateCcw,
  Shield,
  Sun,
  Users,
  X,
} from 'lucide-react'
import { Dashboard } from './components/Dashboard/Dashboard'
import { LoadingState } from './components/shared/Ui'
import { useDashboardData } from './hooks/useDashboardData'
import { API_BASE_URL, api } from './services/api'
import type { SimulationRequest, SimulationSummary } from './types'
import { dateTime } from './utils/format'

type Theme = 'dark' | 'light'

const navItems = [
  { label: 'Overview', href: '#top', icon: Activity },
  { label: 'Rounds', href: '#rounds', icon: Database },
  { label: 'Clients', href: '#clients', icon: Users },
  { label: 'Recovery', href: '#recovery', icon: RotateCcw },
  { label: 'Audit log', href: '#audit', icon: ClipboardList },
]

export default function App() {
  const [runs, setRuns] = useState<SimulationSummary[]>([])
  const [selectedRun, setSelectedRun] = useState<string | null>(null)
  const [theme, setTheme] = useState<Theme>(() => localStorage.getItem('fedsentinel-theme') === 'light' ? 'light' : 'dark')
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [starting, setStarting] = useState(false)
  const [startupError, setStartupError] = useState<string | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [apiOnline, setApiOnline] = useState<boolean | null>(null)
  const dashboard = useDashboardData(selectedRun, autoRefresh)

  const loadRuns = async (preferRun?: string) => {
    try {
      const next = await api.listSimulations()
      setRuns(next)
      setApiOnline(true)
      setStartupError(null)
      setSelectedRun((current) => preferRun || current || next[0]?.run_id || null)
    } catch (caught) {
      setApiOnline(false)
      setStartupError(caught instanceof Error ? caught.message : 'Unable to connect to the backend')
    }
  }

  useEffect(() => {
    void api.health().then(() => setApiOnline(true)).catch(() => setApiOnline(false))
    void loadRuns()
  }, [])

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem('fedsentinel-theme', theme)
  }, [theme])

  const startSimulation = async (request: SimulationRequest) => {
    setStarting(true)
    setStartupError(null)
    try {
      const created = await api.createSimulation(request)
      await loadRuns(created.run_id)
      setSelectedRun(created.run_id)
    } catch (caught) {
      setStartupError(caught instanceof Error ? caught.message : 'Simulation could not be started')
    } finally {
      setStarting(false)
    }
  }

  return (
    <div className="app-shell" id="top">
      <aside className={sidebarOpen ? 'sidebar open' : 'sidebar'}>
        <div className="brand"><span><Shield aria-hidden="true" /></span><div><strong>FedSentinel</strong><small>Defense intelligence</small></div><button type="button" className="mobile-close" onClick={() => setSidebarOpen(false)} aria-label="Close navigation"><X /></button></div>
        <nav aria-label="Dashboard sections">
          <span className="nav-label">Workspace</span>
          {navItems.map((item, index) => <a key={item.href} href={item.href} className={index === 0 ? 'active' : ''} onClick={() => setSidebarOpen(false)}><item.icon aria-hidden="true" />{item.label}</a>)}
        </nav>
        <div className="sidebar-foot">
          <div className="connection-card">
            <span className={'connection-dot ' + (apiOnline ? 'online' : apiOnline === false ? 'offline' : '')} />
            <div><strong>{apiOnline ? 'Backend connected' : apiOnline === false ? 'Backend offline' : 'Checking backend'}</strong><small>{API_BASE_URL}</small></div>
          </div>
          <p>Schema-aware analytics<br />No generated demo values</p>
        </div>
      </aside>

      <div className="workspace">
        <header className="topbar">
          <button type="button" className="menu-button" onClick={() => setSidebarOpen(true)} aria-label="Open navigation"><Menu /></button>
          <div className="run-picker">
            <label htmlFor="run-select">Active run</label>
            <select id="run-select" value={selectedRun || ''} onChange={(event) => setSelectedRun(event.target.value || null)}>
              {!runs.length && <option value="">No simulations yet</option>}
              {runs.map((run) => <option key={run.run_id} value={run.run_id}>{run.run_id} · {run.scenario.replaceAll('_', ' ')}</option>)}
            </select>
          </div>
          <div className="sync-meta">
            <span className={dashboard.refreshing ? 'pulse' : ''} />
            <div><strong>{autoRefresh ? 'Live refresh on' : 'Live refresh paused'}</strong><small>{dashboard.lastUpdated ? 'Synced ' + dateTime(dashboard.lastUpdated.toISOString()) : 'Awaiting first sync'}</small></div>
          </div>
          <label className="auto-toggle"><input type="checkbox" checked={autoRefresh} onChange={(event) => setAutoRefresh(event.target.checked)} /><span>Auto</span></label>
          <button type="button" className="theme-button" onClick={() => setTheme((current) => current === 'dark' ? 'light' : 'dark')} aria-label={'Use ' + (theme === 'dark' ? 'light' : 'dark') + ' theme'}>{theme === 'dark' ? <Sun /> : <Moon />}</button>
        </header>

        {(startupError || dashboard.error) && <div className="error-banner" role="alert"><PanelLeftClose aria-hidden="true" /><div><strong>Backend request failed</strong><span>{startupError || dashboard.error}</span></div><button type="button" onClick={() => { setStartupError(null); void loadRuns(); void dashboard.refresh(false) }}>Retry</button></div>}

        {dashboard.loading && selectedRun
          ? <LoadingState />
          : <Dashboard data={dashboard.data} refreshing={dashboard.refreshing} onRefresh={() => void dashboard.refresh(false)} onStart={startSimulation} starting={starting} />}
      </div>
      {sidebarOpen && <button type="button" className="sidebar-backdrop" onClick={() => setSidebarOpen(false)} aria-label="Close navigation overlay" />}
    </div>
  )
}
