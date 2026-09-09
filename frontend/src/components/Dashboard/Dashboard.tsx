import { Activity, AlertTriangle, Crosshair, Download, RefreshCw, RotateCcw, ShieldCheck, TrendingUp, Users } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import type { DashboardData } from '../../types'
import { exportAnalyticsCsv, exportJson, exportRecoveryCsv } from '../../utils/export'
import { decimal, percent } from '../../utils/format'
import { AuditLog } from '../AuditLog/AuditLog'
import { ImpactPanel } from '../ImpactPanel/ImpactPanel'
import { ModelMetrics } from '../ModelMetrics/ModelMetrics'
import { RoundMonitor } from '../RoundMonitor/RoundMonitor'
import { ThreatPanel } from '../ThreatPanel/ThreatPanel'
import { FederationNetwork } from '../shared/FederationNetwork'
import { MetricCard, PageHeader, StatusBadge } from '../shared/Ui'

export function Dashboard({ data, refreshing, onRefresh, onNavigate }: { data: DashboardData; refreshing: boolean; onRefresh: () => void; onNavigate: (path: string) => void }) {
  const [selectedRound, setSelectedRound] = useState<number | null>(null)
  useEffect(() => setSelectedRound(data.run?.current_round || data.rounds.at(-1)?.round_id || null), [data.run?.run_id, data.run?.current_round, data.rounds])
  const roundThreats = useMemo(() => selectedRound ? data.threats.filter((item) => item.round_id === selectedRound) : [], [data.threats, selectedRound])
  const roundImpacts = useMemo(() => selectedRound ? data.impacts.filter((item) => item.round_id === selectedRound) : [], [data.impacts, selectedRound])
  const latest = data.metrics.at(-1)
  const suspicious = roundThreats.filter((item) => item.threat_level === 'SUSPICIOUS').length
  const malicious = roundThreats.filter((item) => item.threat_level === 'MALICIOUS').length
  const runId = data.run?.run_id
  const actions = <><button className="secondary-button" onClick={onRefresh} disabled={!runId || refreshing}><RefreshCw className={refreshing ? 'spin' : ''} />Refresh</button><div className="export-menu"><button className="primary-button" disabled={!runId}><Download />Export</button>{runId && <div className="export-options"><button onClick={() => exportAnalyticsCsv(data, runId)}>Analytics CSV</button><button onClick={() => exportRecoveryCsv(data, runId)}>Recovery CSV</button><button onClick={() => exportJson(data, runId)}>Complete JSON</button></div>}</div></>

  return <main className="page"><PageHeader eyebrow="Federated defense analytics" title="Security Operations Center" description="Live operational truth from the FedSentinel backend." actions={actions} />
    <section className="run-status-strip"><div><span>Run state</span><StatusBadge value={data.run?.status} /></div><div><span>Round</span><strong>{data.run ? `${data.run.current_round} / ${data.run.round_count}` : '—'}</strong></div><div><span>Clients</span><strong>{data.run?.client_count ?? '—'}</strong></div><div><span>Scenario</span><strong>{data.run?.scenario?.replaceAll('_', ' ') || '—'}</strong></div><div><span>Suspicious</span><strong className="semantic-warning">{roundThreats.length ? suspicious : '—'}</strong></div><div><span>Malicious</span><strong className="semantic-danger">{roundThreats.length ? malicious : '—'}</strong></div></section>
    <section className="metric-grid metric-grid-five"><MetricCard label="Model accuracy" value={percent(latest?.accuracy)} context={latest ? `Round ${latest.round_id} evaluation` : 'No evaluation yet'} tone="success" icon={<TrendingUp />} /><MetricCard label="Model loss" value={decimal(latest?.loss)} context="Persisted model evaluation" icon={<Activity />} /><MetricCard label="Attack success" value={percent(latest?.attack_success_rate)} context={latest?.attack_success_rate == null ? 'Not available' : 'Accepted attacked updates'} tone="warning" icon={<Crosshair />} /><MetricCard label="Detected threats" value={roundThreats.length ? String(suspicious + malicious) : '—'} context="Suspicious and malicious" tone={malicious ? 'danger' : 'neutral'} icon={<AlertTriangle />} /><MetricCard label="Recovery events" value={data.metrics.length ? String(data.metrics.reduce((sum, item) => sum + item.recovery_count, 0)) : '—'} context="Across selected run" tone="accent" icon={<RotateCcw />} /></section>
    <div className="content-grid overview-grid"><FederationNetwork clients={data.clients} threats={data.threats} onSelect={() => onNavigate('/clients')} /><RoundMonitor run={data.run} rounds={data.rounds} selectedRound={selectedRound} onSelectRound={setSelectedRound} /></div>
    <ModelMetrics metrics={data.metrics} />
    <div className="content-grid two-column"><ThreatPanel threats={roundThreats} /><ImpactPanel impacts={roundImpacts} /></div>
    <div className="overview-footer"><button className="insight-link" onClick={() => onNavigate('/clients')}><Users /><span><strong>Investigate clients</strong><small>Open per-client evidence and history</small></span></button><button className="insight-link" onClick={() => onNavigate('/recovery')}><ShieldCheck /><span><strong>Review recovery</strong><small>Inspect selective restoration decisions</small></span></button></div>
    <AuditLog events={data.audit.slice(-6)} />
  </main>
}
