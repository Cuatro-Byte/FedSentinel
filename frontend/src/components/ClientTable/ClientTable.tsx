import { ChevronRight, Search, SlidersHorizontal } from 'lucide-react'
import { useMemo, useState } from 'react'
import type { ClientAnalytics, ClientState, ImpactResult, ThreatResult } from '../../types'
import { decimal, shortId } from '../../utils/format'
import { Drawer, EmptyState, Panel, StatusBadge } from '../shared/Ui'
import { LineChart } from '../shared/LineChart'

type SortKey = 'threat' | 'reputation' | 'round'

export function ClientTable({ clients, threats, impacts }: { clients: ClientState[]; threats: ThreatResult[]; impacts: ImpactResult[] }) {
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState('ALL')
  const [sort, setSort] = useState<SortKey>('threat')
  const [selected, setSelected] = useState<ClientAnalytics | null>(null)
  const rows = useMemo<ClientAnalytics[]>(() => {
    const threatByClient = new Map<string, ThreatResult>()
    threats.forEach((item) => { const current = threatByClient.get(item.client_id); if (!current || item.round_id >= current.round_id) threatByClient.set(item.client_id, item) })
    const impactByUpdate = new Map(impacts.map((item) => [item.update_id, item]))
    return clients.map((client) => { const threat = threatByClient.get(client.client_id) || null; return { ...client, threat, impact: threat ? impactByUpdate.get(threat.update_id) || null : null } })
      .filter((client) => client.client_id.toLowerCase().includes(search.toLowerCase()))
      .filter((client) => filter === 'ALL' || client.threat?.threat_level === filter)
      .sort((a, b) => sort === 'reputation' ? (a.threat?.reputation_score ?? a.reputation_score) - (b.threat?.reputation_score ?? b.reputation_score) : sort === 'round' ? (b.threat?.round_id ?? 0) - (a.threat?.round_id ?? 0) : (b.threat?.threat_score ?? b.latest_threat_score ?? -1) - (a.threat?.threat_score ?? a.latest_threat_score ?? -1))
  }, [clients, threats, impacts, search, filter, sort])

  return <><Panel title="Client intelligence" eyebrow="Per-client evidence" className="client-panel" action={<span className="table-count">{rows.length} clients</span>}>
    <div className="table-toolbar"><label className="search-field"><Search /><span className="sr-only">Search clients</span><input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search client ID" /></label><label className="filter-field"><SlidersHorizontal /><span className="sr-only">Filter classification</span><select value={filter} onChange={(e) => setFilter(e.target.value)}><option value="ALL">All classifications</option><option value="SAFE">Safe</option><option value="SUSPICIOUS">Suspicious</option><option value="MALICIOUS">Malicious</option></select></label><label className="sort-field"><span className="sr-only">Sort clients</span><select value={sort} onChange={(e) => setSort(e.target.value as SortKey)}><option value="threat">Highest threat</option><option value="reputation">Lowest reputation</option><option value="round">Latest round</option></select></label></div>
    {!clients.length ? <EmptyState title="No client state" description="Client records from the selected run will appear here." /> : <div className="table-scroll"><table><thead><tr><th>Client</th><th>Classification</th><th>Threat</th><th>Impact</th><th>Reputation</th><th>Latest round</th><th>Action</th><th><span className="sr-only">Inspect</span></th></tr></thead><tbody>{rows.map((client) => <tr key={client.client_id} className="clickable-row" onClick={() => setSelected(client)}><td><span className="client-id"><span>{client.client_id.slice(-2).toUpperCase()}</span><strong>{client.client_id}</strong></span></td><td><StatusBadge value={client.threat?.threat_level} /></td><td><Score value={client.threat?.threat_score ?? client.latest_threat_score} /></td><td><Score value={client.impact?.impact_score ?? client.latest_impact_score} /></td><td><Score value={client.threat?.reputation_score ?? client.reputation_score} positive /></td><td>R{client.threat?.round_id ?? '—'}</td><td><StatusBadge value={client.threat?.action || client.action} /></td><td><button className="icon-button" aria-label={`Inspect ${client.client_id}`} onClick={(e) => { e.stopPropagation(); setSelected(client) }}><ChevronRight /></button></td></tr>)}</tbody></table></div>}
  </Panel><ClientDrawer client={selected} threats={threats} impacts={impacts} onClose={() => setSelected(null)} /></>
}

function ClientDrawer({ client, threats, impacts, onClose }: { client: ClientAnalytics | null; threats: ThreatResult[]; impacts: ImpactResult[]; onClose: () => void }) {
  if (!client) return null
  const history = threats.filter((item) => item.client_id === client.client_id).sort((a, b) => a.round_id - b.round_id)
  const impactByUpdate = new Map(impacts.map((item) => [item.update_id, item]))
  const current = history.at(-1) || client.threat
  const currentImpact = current ? impactByUpdate.get(current.update_id) : client.impact
  return <Drawer open title={client.client_id} subtitle="Client evidence" onClose={onClose}><div className="drawer-status"><StatusBadge value={current?.threat_level} /><StatusBadge value={current?.action || client.action} /></div><div className="definition-grid"><div><dt>Threat score</dt><dd>{decimal(current?.threat_score ?? client.latest_threat_score)}</dd></div><div><dt>Reputation</dt><dd>{decimal(current?.reputation_score ?? client.reputation_score)}</dd></div><div><dt>Anomaly</dt><dd>{decimal(current?.anomaly_score)}</dd></div><div><dt>Similarity</dt><dd>{decimal(current?.similarity_score)}</dd></div><div><dt>Estimated impact</dt><dd>{decimal(currentImpact?.impact_score ?? client.latest_impact_score)}</dd></div><div><dt>Latest round</dt><dd>{current ? `R${current.round_id}` : '—'}</dd></div><div className="wide"><dt>Update ID</dt><dd>{shortId(current?.update_id, 28)}</dd></div></div>{history.length > 1 && <section className="drawer-section"><h3>Threat history</h3><LineChart labels={history.map((item) => item.round_id)} series={[{ label: 'Threat', values: history.map((item) => item.threat_score), className: 'line-danger' }, { label: 'Reputation', values: history.map((item) => item.reputation_score), className: 'line-safe' }]} valueFormatter={(value) => value.toFixed(2)} ariaLabel={`Threat and reputation history for ${client.client_id}`} /></section>}<section className="drawer-section"><h3>Decision evidence</h3>{current?.explanation_codes?.length ? <ul className="evidence-list">{current.explanation_codes.map((item) => <li key={item}>{item.replaceAll('_', ' ')}</li>)}</ul> : <p className="muted-copy">No explanation codes were provided by the backend.</p>}</section></Drawer>
}

function Score({ value, positive = false }: { value: number | null | undefined; positive?: boolean }) {
  if (value == null) return <span className="not-available">—</span>
  const safe = Math.max(0, Math.min(1, value)); const tone = positive ? 'score-positive' : safe >= 0.8 ? 'score-danger' : safe >= 0.5 ? 'score-warning' : ''
  return <div className="score-bar"><span>{decimal(value)}</span><i><b style={{ width: `${safe * 100}%` }} className={tone} /></i></div>
}
