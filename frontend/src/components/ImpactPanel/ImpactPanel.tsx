import type { ImpactLevel, ImpactResult } from '../../types'
import { decimal, percent } from '../../utils/format'
import { EmptyState, Panel } from '../shared/Ui'

const levels: ImpactLevel[] = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']

export function ImpactPanel({ impacts }: { impacts: ImpactResult[] }) {
  const average = impacts.length ? impacts.reduce((sum, item) => sum + item.impact_score, 0) / impacts.length : null
  const maxInfluence = impacts.reduce<ImpactResult | null>((top, item) =>
    !top || (item.influence_estimate ?? 0) > (top.influence_estimate ?? 0) ? item : top, null)
  return (
    <Panel title="Estimated impact" eyebrow="Influence analysis" className="impact-panel">
      {!impacts.length ? <EmptyState title="No impact estimates" description="Impact calculations for the selected round will appear here." /> : (
        <div className="impact-layout">
          <div className="impact-summary"><div><span>Average impact</span><strong>{percent(average)}</strong><small>Selected round</small></div><div><span>Highest influence</span><strong>{decimal(maxInfluence?.influence_estimate)}</strong><small>{maxInfluence?.client_id}</small></div></div>
          <div className="impact-bars">
            {levels.map((level) => {
              const count = impacts.filter((item) => item.impact_level === level).length
              const value = count / impacts.length * 100
              return <div key={level}><div><span>{level}</span><b>{count}</b></div><div className="impact-track"><span className={'level-' + level.toLowerCase()} style={{ width: value + '%' }} /></div></div>
            })}
          </div>
        </div>
      )}
    </Panel>
  )
}
