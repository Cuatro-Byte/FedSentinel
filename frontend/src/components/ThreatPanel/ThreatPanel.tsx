import { ShieldCheck, ShieldQuestion, ShieldX } from 'lucide-react'
import type { ThreatResult } from '../../types'
import { percent } from '../../utils/format'
import { EmptyState, Panel } from '../shared/Ui'

export function ThreatPanel({ threats }: { threats: ThreatResult[] }) {
  const counts = {
    SAFE: threats.filter((item) => item.threat_level === 'SAFE').length,
    SUSPICIOUS: threats.filter((item) => item.threat_level === 'SUSPICIOUS').length,
    MALICIOUS: threats.filter((item) => item.threat_level === 'MALICIOUS').length,
  }
  const total = threats.length
  const safeAngle = total ? counts.SAFE / total * 360 : 0
  const suspiciousAngle = total ? (counts.SAFE + counts.SUSPICIOUS) / total * 360 : 0

  return (
    <Panel title="Threat posture" eyebrow="Detection" className="threat-panel">
      {!total ? <EmptyState title="No threat results" description="Sentinel results for the selected round will appear here." /> : (
        <div className="threat-layout">
          <div className="donut-wrap">
            <div className="donut" style={{ '--safe-angle': safeAngle + 'deg', '--suspicious-angle': suspiciousAngle + 'deg' } as React.CSSProperties}>
              <div><strong>{total}</strong><span>updates</span></div>
            </div>
          </div>
          <div className="distribution-list">
            <div><span className="distribution-icon safe"><ShieldCheck aria-hidden="true" /></span><span><strong>Safe</strong><small>{percent(counts.SAFE / total)}</small></span><b>{counts.SAFE}</b></div>
            <div><span className="distribution-icon suspicious"><ShieldQuestion aria-hidden="true" /></span><span><strong>Suspicious</strong><small>{percent(counts.SUSPICIOUS / total)}</small></span><b>{counts.SUSPICIOUS}</b></div>
            <div><span className="distribution-icon malicious"><ShieldX aria-hidden="true" /></span><span><strong>Malicious</strong><small>{percent(counts.MALICIOUS / total)}</small></span><b>{counts.MALICIOUS}</b></div>
          </div>
        </div>
      )}
    </Panel>
  )
}
