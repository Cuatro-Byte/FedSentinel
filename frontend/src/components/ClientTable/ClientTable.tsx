import { ChevronDown, Search, SlidersHorizontal } from 'lucide-react'
import { useMemo, useState } from 'react'
import type { ClientAnalytics, ClientState, ImpactResult, ThreatResult } from '../../types'
import { decimal } from '../../utils/format'
import { EmptyState, Panel, StatusBadge } from '../shared/Ui'

export function ClientTable({ clients, threats, impacts }: { clients: ClientState[]; threats: ThreatResult[]; impacts: ImpactResult[] }) {
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState('ALL')
  const [expanded, setExpanded] = useState<string | null>(null)
  const rows = useMemo<ClientAnalytics[]>(() => {
    const threatByClient = new Map(threats.map((item) => [item.client_id, item]))
    const impactByClient = new Map(impacts.map((item) => [item.client_id, item]))
    return clients
      .map((client) => ({ ...client, threat: threatByClient.get(client.client_id) || null, impact: impactByClient.get(client.client_id) || null }))
      .filter((client) => client.client_id.toLowerCase().includes(search.toLowerCase()))
      .filter((client) => filter === 'ALL' || client.threat?.threat_level === filter)
      .sort((a, b) => (b.threat?.threat_score ?? -1) - (a.threat?.threat_score ?? -1))
  }, [clients, threats, impacts, search, filter])

  return (
    <Panel id="clients" title="Client intelligence" eyebrow="Per-client evidence" className="client-panel" action={<span className="table-count">{rows.length} clients</span>}>
      <div className="table-toolbar">
        <label className="search-field"><Search aria-hidden="true" /><span className="sr-only">Search clients</span><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search client ID" /></label>
        <label className="filter-field"><SlidersHorizontal aria-hidden="true" /><span className="sr-only">Filter threat level</span><select value={filter} onChange={(event) => setFilter(event.target.value)}><option value="ALL">All levels</option><option value="SAFE">Safe</option><option value="SUSPICIOUS">Suspicious</option><option value="MALICIOUS">Malicious</option></select></label>
      </div>
      {!clients.length ? <EmptyState title="No client state" description="Client records from the selected run will appear here." /> : (
        <div className="table-scroll">
          <table>
            <thead><tr><th>Client</th><th>Status</th><th>Threat</th><th>Impact</th><th>Reputation</th><th>Action</th><th><span className="sr-only">Details</span></th></tr></thead>
            <tbody>{rows.map((client) => <ClientRow key={client.client_id} client={client} expanded={expanded === client.client_id} onToggle={() => setExpanded((value) => value === client.client_id ? null : client.client_id)} />)}</tbody>
          </table>
        </div>
      )}
    </Panel>
  )
}

function ClientRow({ client, expanded, onToggle }: { client: ClientAnalytics; expanded: boolean; onToggle: () => void }) {
  const reason = client.threat?.explanation_codes?.join(', ') || 'No explanation code provided'
  return (
    <>
      <tr>
        <td><span className="client-id"><span>{client.client_id.slice(-2).toUpperCase()}</span><strong>{client.client_id}</strong></span></td>
        <td><StatusBadge value={client.threat?.threat_level || 'PENDING'} /></td>
        <td><Score value={client.threat?.threat_score ?? client.latest_threat_score} /></td>
        <td><Score value={client.impact?.impact_score ?? client.latest_impact_score} /></td>
        <td><Score value={client.threat?.reputation_score ?? client.reputation_score} positive /></td>
        <td><StatusBadge value={client.threat?.action || client.action || 'PENDING'} /></td>
        <td><button type="button" className="icon-button" aria-label={'Toggle details for ' + client.client_id} aria-expanded={expanded} onClick={onToggle}><ChevronDown className={expanded ? 'expanded' : ''} aria-hidden="true" /></button></td>
      </tr>
      {expanded && <tr className="detail-row"><td colSpan={7}><div className="client-details"><div><span>Update ID</span><strong>{client.threat?.update_id || '—'}</strong></div><div><span>Anomaly</span><strong>{decimal(client.threat?.anomaly_score)}</strong></div><div><span>Similarity</span><strong>{decimal(client.threat?.similarity_score)}</strong></div><div><span>Influence</span><strong>{decimal(client.impact?.influence_estimate)}</strong></div><div className="wide"><span>Reason</span><strong>{reason}</strong></div></div></td></tr>}
    </>
  )
}

function Score({ value, positive = false }: { value: number | null | undefined; positive?: boolean }) {
  const safe = Math.max(0, Math.min(1, value ?? 0))
  const tone = positive ? 'score-positive' : safe >= 0.8 ? 'score-danger' : safe >= 0.5 ? 'score-warning' : ''
  return <div className="score-bar"><span>{decimal(value)}</span><i><b style={{ width: safe * 100 + '%' }} className={tone} /></i></div>
}
