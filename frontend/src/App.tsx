import { useCallback, useEffect, useMemo, useState } from 'react'
import { Activity, BarChart3, ChevronLeft, ChevronRight, ClipboardList, Database, FlaskConical, Menu, Moon, Play, RefreshCw, RotateCcw, Shield, ShieldAlert, Sun, Users, X } from 'lucide-react'
import { Dashboard } from './components/Dashboard/Dashboard'
import { LoadingState, StatusBadge } from './components/shared/Ui'
import { useDashboardData } from './hooks/useDashboardData'
import { API_BASE_URL, api } from './services/api'
import type { SimulationRequest, SimulationSummary } from './types'
import type { AppRoute } from './types/navigation'
import { dateTime } from './utils/format'
import { AuditPage, ClientsPage, ImpactPage, NoRunPage, RecoveryPage, RoundsPage, SimulationPage, ThreatsPage, ValidationPage } from './pages/Pages'

type Theme = 'dark' | 'light'
const validRoutes: AppRoute[] = ['/dashboard', '/simulation', '/rounds', '/clients', '/threats', '/impact', '/recovery', '/validation', '/audit']
const navGroups = [
  { label: 'Workspace', items: [{ label: 'Overview', path: '/dashboard', icon: Activity }, { label: 'Simulation', path: '/simulation', icon: Play }, { label: 'Rounds', path: '/rounds', icon: Database }, { label: 'Clients', path: '/clients', icon: Users }] },
  { label: 'Security', items: [{ label: 'Threat analysis', path: '/threats', icon: ShieldAlert }, { label: 'Impact analysis', path: '/impact', icon: BarChart3 }, { label: 'Recovery', path: '/recovery', icon: RotateCcw }, { label: 'Validation', path: '/validation', icon: FlaskConical }] },
  { label: 'System', items: [{ label: 'Audit log', path: '/audit', icon: ClipboardList }] },
] as const

function normalizeRoute(path: string): AppRoute { return validRoutes.includes(path as AppRoute) ? path as AppRoute : '/dashboard' }

export default function App() {
  const [route, setRoute] = useState<AppRoute>(() => normalizeRoute(window.location.pathname))
  const [runs, setRuns] = useState<SimulationSummary[]>([])
  const [selectedRun, setSelectedRun] = useState<string | null>(null)
  const [theme, setTheme] = useState<Theme>(() => localStorage.getItem('fedsentinel-theme') === 'light' ? 'light' : 'dark')
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [starting, setStarting] = useState(false)
  const [startupError, setStartupError] = useState<string | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem('fedsentinel-sidebar') === 'collapsed')
  const [apiOnline, setApiOnline] = useState<boolean | null>(null)
  const dashboard = useDashboardData(selectedRun, autoRefresh)

  const navigate = useCallback((path: string) => { const next = normalizeRoute(path); window.history.pushState({}, '', next); setRoute(next); setSidebarOpen(false); window.scrollTo({ top: 0, behavior: 'smooth' }) }, [])
  const loadRuns = useCallback(async (preferRun?: string) => { try { const next = await api.listSimulations(); setRuns(next); setApiOnline(true); setStartupError(null); setSelectedRun((current) => preferRun || current || next[0]?.run_id || null) } catch (caught) { setApiOnline(false); setStartupError(caught instanceof Error ? caught.message : 'Unable to connect to the backend') } }, [])

  useEffect(() => { if (window.location.pathname !== route) window.history.replaceState({}, '', route); const onPop = () => setRoute(normalizeRoute(window.location.pathname)); window.addEventListener('popstate', onPop); void api.health().then(() => setApiOnline(true)).catch(() => setApiOnline(false)); void loadRuns(); return () => window.removeEventListener('popstate', onPop) }, [loadRuns])
  useEffect(() => { document.documentElement.dataset.theme = theme; localStorage.setItem('fedsentinel-theme', theme) }, [theme])
  useEffect(() => { localStorage.setItem('fedsentinel-sidebar', collapsed ? 'collapsed' : 'open') }, [collapsed])
  useEffect(() => { if (!autoRefresh) return; const timer = window.setInterval(() => void loadRuns(), 4000); return () => clearInterval(timer) }, [autoRefresh, loadRuns])

  const startSimulation = async (request: SimulationRequest) => { setStarting(true); setStartupError(null); try { const created = await api.createSimulation(request); await loadRuns(created.run_id); setSelectedRun(created.run_id); navigate('/dashboard') } catch (caught) { setStartupError(caught instanceof Error ? caught.message : 'Simulation could not be started') } finally { setStarting(false) } }
  const selectedSummary = runs.find((run) => run.run_id === selectedRun)
  const page = useMemo(() => {
    if (!selectedRun && route !== '/simulation' && route !== '/validation') return <NoRunPage onNavigate={navigate} />
    switch (route) {
      case '/simulation': return <SimulationPage data={dashboard.data} onStart={startSimulation} starting={starting} />
      case '/rounds': return <RoundsPage data={dashboard.data} />
      case '/clients': return <ClientsPage data={dashboard.data} />
      case '/threats': return <ThreatsPage data={dashboard.data} />
      case '/impact': return <ImpactPage data={dashboard.data} />
      case '/recovery': return <RecoveryPage data={dashboard.data} />
      case '/validation': return <ValidationPage />
      case '/audit': return <AuditPage data={dashboard.data} />
      default: return <Dashboard data={dashboard.data} refreshing={dashboard.refreshing} onRefresh={() => void dashboard.refresh(false)} onNavigate={navigate} />
    }
  }, [route, selectedRun, dashboard.data, dashboard.refreshing, starting, navigate])

  return <div className={`app-shell ${collapsed ? 'sidebar-collapsed' : ''}`}>
    <aside className={sidebarOpen ? 'sidebar open' : 'sidebar'}><div className="brand"><span><Shield /></span><div><strong>FedSentinel</strong><small>Defense Intelligence</small></div><button className="mobile-close" onClick={() => setSidebarOpen(false)} aria-label="Close navigation"><X /></button></div><nav aria-label="Primary navigation">{navGroups.map((group) => <div className="nav-group" key={group.label}><span className="nav-label">{group.label}</span>{group.items.map((item) => <button title={collapsed ? item.label : undefined} key={item.path} className={route === item.path ? 'active' : ''} onClick={() => navigate(item.path)}><item.icon /><span>{item.label}</span></button>)}</div>)}</nav><button className="collapse-button" onClick={() => setCollapsed((value) => !value)} aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}>{collapsed ? <ChevronRight /> : <><ChevronLeft /><span>Collapse sidebar</span></>}</button><div className="sidebar-foot"><div className="connection-card"><span className={`connection-dot ${apiOnline ? 'online' : apiOnline === false ? 'offline' : ''}`} /><div><strong>{apiOnline ? 'Backend connected' : apiOnline === false ? 'Backend offline' : 'Checking backend'}</strong><small>{API_BASE_URL}</small></div></div></div></aside>
    <div className="workspace"><header className="topbar"><button className="menu-button" onClick={() => setSidebarOpen(true)} aria-label="Open navigation"><Menu /></button><div className="run-picker"><label htmlFor="run-select">Active run</label><div><select id="run-select" value={selectedRun || ''} onChange={(e) => setSelectedRun(e.target.value || null)}><option value="">No simulation selected</option>{runs.map((run) => <option key={run.run_id} value={run.run_id}>{run.run_id} · {run.scenario.replaceAll('_', ' ')}</option>)}</select>{selectedSummary && <StatusBadge value={selectedSummary.status} />}</div></div><div className="sync-meta"><span className={dashboard.refreshing ? 'pulse' : ''} /><div><strong>{autoRefresh ? 'Live refresh on' : 'Live refresh paused'}</strong><small>{dashboard.lastUpdated ? `Synced ${dateTime(dashboard.lastUpdated.toISOString())}` : 'Awaiting first sync'}</small></div></div><label className="auto-toggle"><input type="checkbox" checked={autoRefresh} onChange={(e) => setAutoRefresh(e.target.checked)} /><span>Auto</span></label><button className="theme-button" onClick={() => setTheme((current) => current === 'dark' ? 'light' : 'dark')} aria-label={`Use ${theme === 'dark' ? 'light' : 'dark'} theme`}>{theme === 'dark' ? <Sun /> : <Moon />}</button></header>
      {(startupError || dashboard.error) && <div className="error-banner" role="alert"><ShieldAlert /><div><strong>Backend request failed</strong><span>{startupError || dashboard.error}</span></div><button onClick={() => { setStartupError(null); void loadRuns(); void dashboard.refresh(false) }}><RefreshCw />Retry</button></div>}
      {dashboard.loading && selectedRun ? <LoadingState /> : page}
    </div>{sidebarOpen && <button className="sidebar-backdrop" onClick={() => setSidebarOpen(false)} aria-label="Close navigation overlay" />}
  </div>
}
