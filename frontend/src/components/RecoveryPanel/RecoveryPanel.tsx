import { ArrowRight, Download, RotateCcw, ShieldCheck } from 'lucide-react'
import type { RecoveryRecord } from '../../types'
import { decimal, shortId } from '../../utils/format'
import { EmptyState, Panel, StatusBadge } from '../shared/Ui'

export function RecoveryPanel({ recoveries, onExport }: { recoveries: RecoveryRecord[]; onExport: () => void }) {
  const relevant = recoveries.filter((item) => item.recovery_status !== 'NOT_REQUIRED')
  const latest = relevant.at(-1)
  return (
    <Panel
      id="recovery"
      title="Selective recovery"
      eyebrow="Model restoration"
      className="recovery-panel"
      action={<button className="secondary-button compact" type="button" onClick={onExport} disabled={!recoveries.length}><Download aria-hidden="true" /> CSV</button>}
    >
      {!latest ? <EmptyState title="No recovery required" description={recoveries.length ? 'The backend evaluated every completed round without recording a recovery event.' : 'Recovery evaluations will appear after a simulation runs.'} /> : (
        <>
          <div className="recovery-callout"><span><RotateCcw aria-hidden="true" /></span><div><StatusBadge value={latest.recovery_status} /><strong>Recovery evaluated in round {latest.round_id}</strong><small>{latest.trigger}</small></div></div>
          <div className="recovery-models">
            <div><small>Previous model</small><strong>{shortId(latest.previous_model_version, 24)}</strong></div>
            <ArrowRight aria-hidden="true" />
            <div><small>Recovered model</small><strong>{shortId(latest.recovered_model_version, 24)}</strong></div>
          </div>
          <div className="recovery-grid">
            <div><span>Accuracy</span><strong>{decimal(latest.before_accuracy)} <ArrowRight aria-hidden="true" /> {decimal(latest.after_accuracy)}</strong></div>
            <div><span>Loss</span><strong>{decimal(latest.before_loss)} <ArrowRight aria-hidden="true" /> {decimal(latest.after_loss)}</strong></div>
            <div><span>Excluded</span><strong>{latest.excluded_client_ids.length} clients</strong></div>
            <div><span>Affected</span><strong>{latest.affected_update_ids.length} updates</strong></div>
          </div>
          {!!latest.excluded_client_ids.length && <div className="excluded-clients"><ShieldCheck aria-hidden="true" /><span>{latest.excluded_client_ids.join(', ')}</span></div>}
        </>
      )}
    </Panel>
  )
}
