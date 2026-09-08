import { Box, CheckCircle2, Clock3, GitCommitHorizontal, UsersRound } from 'lucide-react'
import type { FLRound, SimulationRun } from '../../types'
import { shortId } from '../../utils/format'
import { EmptyState, Panel, StatusBadge } from '../shared/Ui'

export function RoundMonitor({ run, rounds, selectedRound, onSelectRound }: {
  run: SimulationRun | null
  rounds: FLRound[]
  selectedRound: number | null
  onSelectRound: (round: number) => void
}) {
  if (!run) return (
    <Panel id="rounds" title="Round monitor" eyebrow="Live pipeline">
      <EmptyState title="No run selected" description="Launch a simulation or select a previous run to inspect its rounds." />
    </Panel>
  )
  const current = rounds.find((round) => round.round_id === selectedRound) || rounds.at(-1)
  const progress = run.round_count ? Math.min(100, (run.current_round / run.round_count) * 100) : 0

  return (
    <Panel id="rounds" title="Round monitor" eyebrow="Live pipeline" action={<StatusBadge value={run.status} />}>
      <div className="round-progress-head">
        <div><strong>{run.current_round} / {run.round_count} rounds</strong><span>{run.scenario.replaceAll('_', ' ')}</span></div>
        <span>{Math.round(progress)}%</span>
      </div>
      <div className="progress-track"><span style={{ width: progress + '%' }} /></div>
      <div className="round-facts">
        <span><UsersRound aria-hidden="true" /> {run.client_count} registered</span>
        <span><Box aria-hidden="true" /> {current?.update_count ?? 0} updates</span>
        <span><GitCommitHorizontal aria-hidden="true" /> {shortId(current?.model_version || run.model_version, 20)}</span>
        <span>{run.status === 'COMPLETED' ? <CheckCircle2 aria-hidden="true" /> : <Clock3 aria-hidden="true" />} {current?.status || run.status}</span>
      </div>
      <div className="round-selector" aria-label="Select a federated round">
        {rounds.map((round) => (
          <button type="button" className={round.round_id === selectedRound ? 'active' : ''} onClick={() => onSelectRound(round.round_id)} key={round.round_id}>
            <span>R{round.round_id}</span>
            <small>{round.quarantined_count ? round.quarantined_count + ' blocked' : 'clear'}</small>
          </button>
        ))}
      </div>
    </Panel>
  )
}
