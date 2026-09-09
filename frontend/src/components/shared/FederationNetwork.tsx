import { Server } from 'lucide-react'
import type { ClientState, ThreatResult } from '../../types'
import { decimal } from '../../utils/format'
import { EmptyState, Panel, StatusBadge } from './Ui'

export function FederationNetwork({
  clients,
  threats,
  onSelect,
}: {
  clients: ClientState[]
  threats: ThreatResult[]
  onSelect?: (clientId: string) => void
}) {
  const latest = new Map<string, ThreatResult>()
  threats.forEach((threat) => {
    const current = latest.get(threat.client_id)
    if (!current || threat.round_id >= current.round_id) latest.set(threat.client_id, threat)
  })

  return (
    <Panel
      title="Federation network"
      eyebrow="Live topology"
      className="network-panel"
      action={<span className="table-count">{clients.length} nodes</span>}
    >
      {!clients.length ? (
        <EmptyState title="No clients connected" description="Select a run to map its federated clients." />
      ) : (
        <div className="network-stage">
          <svg viewBox="0 0 640 360" role="img" aria-label={`Federation network with ${clients.length} clients`}>
            <g className="network-links">
              {clients.map((client, index) => {
                const angle = (index / clients.length) * Math.PI * 2 - Math.PI / 2
                const x = 320 + Math.cos(angle) * 240
                const y = 180 + Math.sin(angle) * 130
                return <line key={client.client_id} x1="320" y1="180" x2={x} y2={y} />
              })}
            </g>
            <g className="server-node">
              <circle cx="320" cy="180" r="48" />
              <foreignObject x="292" y="151" width="56" height="58">
                <div className="server-label">
                  <Server />
                  <span>Global</span>
                </div>
              </foreignObject>
            </g>
            {clients.map((client, index) => {
              const threat = latest.get(client.client_id)
              const level = threat?.threat_level || 'UNKNOWN'
              const angle = (index / clients.length) * Math.PI * 2 - Math.PI / 2
              const x = 320 + Math.cos(angle) * 240
              const y = 180 + Math.sin(angle) * 130
              return (
                <g
                  key={client.client_id}
                  className={`client-node node-${level.toLowerCase()}`}
                  role="button"
                  tabIndex={0}
                  onClick={() => onSelect?.(client.client_id)}
                  onKeyDown={(event) => event.key === 'Enter' && onSelect?.(client.client_id)}
                >
                  <circle cx={x} cy={y} r="19">
                    <title>{`${client.client_id} · ${level} · threat ${decimal(threat?.threat_score ?? client.latest_threat_score)}`}</title>
                  </circle>
                  <text x={x} y={y + 34} textAnchor="middle">
                    {client.client_id}
                  </text>
                </g>
              )
            })}
          </svg>
          <div className="network-legend">
            <StatusBadge value="SAFE" />
            <StatusBadge value="SUSPICIOUS" />
            <StatusBadge value="MALICIOUS" />
            <StatusBadge value="UNKNOWN" />
          </div>
        </div>
      )}
    </Panel>
  )
}
