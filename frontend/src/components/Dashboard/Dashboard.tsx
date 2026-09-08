import { Download, RefreshCw } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import type { DashboardData, SimulationRequest } from '../../types'
import { exportAnalyticsCsv, exportJson, exportRecoveryCsv } from '../../utils/export'
import { AttackControl } from '../AttackControl/AttackControl'
import { AuditLog } from '../AuditLog/AuditLog'
import { ClientTable } from '../ClientTable/ClientTable'
import { ImpactPanel } from '../ImpactPanel/ImpactPanel'
import { ModelMetrics } from '../ModelMetrics/ModelMetrics'
import { RecoveryPanel } from '../RecoveryPanel/RecoveryPanel'
import { RoundMonitor } from '../RoundMonitor/RoundMonitor'
import { ThreatPanel } from '../ThreatPanel/ThreatPanel'

interface DashboardProps {
  data: DashboardData
  refreshing: boolean
  onRefresh: () => void
  onStart: (request: SimulationRequest) => Promise<void>
  starting: boolean
}

export function Dashboard({ data, refreshing, onRefresh, onStart, starting }: DashboardProps) {
  const [selectedRound, setSelectedRound] = useState<number | null>(null)

  useEffect(() => {
    setSelectedRound(data.run?.current_round || data.rounds.at(-1)?.round_id || null)
  }, [data.run?.run_id, data.run?.current_round, data.rounds])

  const roundThreats = useMemo(
    () => selectedRound ? data.threats.filter((item) => item.round_id === selectedRound) : [],
    [data.threats, selectedRound],
  )
  const roundImpacts = useMemo(
    () => selectedRound ? data.impacts.filter((item) => item.round_id === selectedRound) : [],
    [data.impacts, selectedRound],
  )
  const runId = data.run?.run_id

  return (
    <main className="dashboard-main">
      <div className="dashboard-heading">
        <div>
          <span className="eyebrow">Federated defense analytics</span>
          <h1>Security operations center</h1>
          <p>Every value shown is read from the FedSentinel backend.</p>
        </div>
        <div className="heading-actions">
          <button type="button" className="secondary-button" onClick={onRefresh} disabled={!runId || refreshing}>
            <RefreshCw className={refreshing ? 'spin' : ''} aria-hidden="true" /> Refresh
          </button>
          <div className="export-menu">
            <button type="button" className="primary-button" disabled={!runId}>
              <Download aria-hidden="true" /> Export
            </button>
            {runId && <div className="export-options">
              <button type="button" onClick={() => exportAnalyticsCsv(data, runId)}>Analytics CSV</button>
              <button type="button" onClick={() => exportRecoveryCsv(data, runId)}>Recovery CSV</button>
              <button type="button" onClick={() => exportJson(data, runId)}>Complete JSON</button>
            </div>}
          </div>
        </div>
      </div>

      <div className="dashboard-grid top-grid">
        <AttackControl onStart={onStart} running={starting} />
        <RoundMonitor run={data.run} rounds={data.rounds} selectedRound={selectedRound} onSelectRound={setSelectedRound} />
      </div>
      <ModelMetrics metrics={data.metrics} />
      <div className="dashboard-grid analytics-grid">
        <ThreatPanel threats={roundThreats} />
        <ImpactPanel impacts={roundImpacts} />
      </div>
      <ClientTable clients={data.clients} threats={roundThreats} impacts={roundImpacts} />
      <div className="dashboard-grid bottom-grid">
        <RecoveryPanel recoveries={data.recoveries} onExport={() => runId && exportRecoveryCsv(data, runId)} />
        <AuditLog events={data.audit} />
      </div>
    </main>
  )
}
