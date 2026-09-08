import { AlertTriangle, CheckCircle2, Clock3, Info } from 'lucide-react'
import type { AuditEvent } from '../../types'
import { dateTime } from '../../utils/format'
import { EmptyState, Panel } from '../shared/Ui'

export function AuditLog({ events }: { events: AuditEvent[] }) {
  const visible = [...events].reverse().slice(0, 10)
  return (
    <Panel id="audit" title="Security audit trail" eyebrow="Persisted backend events" className="audit-panel" action={<span className="table-count">{events.length} events</span>}>
      {!events.length ? <EmptyState title="No audit events" description="Simulation and mitigation events will be listed here." /> : (
        <ol className="audit-list">
          {visible.map((event) => (
            <li key={event.id}>
              <span className={'audit-icon ' + event.severity.toLowerCase()}>{event.severity === 'ALERT' ? <AlertTriangle aria-hidden="true" /> : event.severity === 'WARN' ? <Info aria-hidden="true" /> : event.event_type.includes('COMPLETED') ? <CheckCircle2 aria-hidden="true" /> : <Clock3 aria-hidden="true" />}</span>
              <div><strong>{event.event_type.replaceAll('_', ' ')}</strong><p>{event.message}</p><small>{dateTime(event.timestamp)}{event.round_id ? ' · Round ' + event.round_id : ''}{event.client_id ? ' · ' + event.client_id : ''}</small></div>
            </li>
          ))}
        </ol>
      )}
    </Panel>
  )
}
