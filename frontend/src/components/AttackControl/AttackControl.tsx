import { useState } from 'react'
import { Play, ShieldAlert } from 'lucide-react'
import type { SimulationRequest } from '../../types'
import { Panel } from '../shared/Ui'

const scenarios = [
  ['normal', 'Normal / no attack'],
  ['model_poisoning', 'Model poisoning'],
  ['label_poisoning', 'Label poisoning'],
  ['backdoor', 'Backdoor'],
  ['mixed_attack', 'Mixed attack'],
  ['sleeper', 'Sleeper attack'],
]

export function AttackControl({ onStart, running }: { onStart: (request: SimulationRequest) => Promise<void>; running: boolean }) {
  const [request, setRequest] = useState<SimulationRequest>({ client_count: 10, rounds: 5, scenario: 'backdoor', attack_enabled: true })
  const update = <K extends keyof SimulationRequest>(key: K, value: SimulationRequest[K]) =>
    setRequest((current) => ({ ...current, [key]: value }))

  return (
    <Panel id="control" title="Launch simulation" eyebrow="Attack control" className="attack-control" action={<ShieldAlert aria-hidden="true" />}>
      <form onSubmit={(event) => { event.preventDefault(); void onStart(request) }}>
        <label>
          <span>Scenario</span>
          <select value={request.scenario} onChange={(event) => update('scenario', event.target.value)}>
            {scenarios.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
        </label>
        <div className="form-row">
          <label>
            <span>Clients</span>
            <input type="number" min="1" max="100" value={request.client_count} onChange={(event) => update('client_count', Number(event.target.value))} />
          </label>
          <label>
            <span>Rounds</span>
            <input type="number" min="1" max="100" value={request.rounds} onChange={(event) => update('rounds', Number(event.target.value))} />
          </label>
        </div>
        <label className="switch-row">
          <span><strong>Attack enabled</strong><small>Ground truth stays isolated from Sentinel.</small></span>
          <input type="checkbox" checked={request.attack_enabled} onChange={(event) => update('attack_enabled', event.target.checked)} />
        </label>
        <button className="primary-button" type="submit" disabled={running}>
          <Play aria-hidden="true" />
          {running ? 'Calculating simulation…' : 'Start simulation'}
        </button>
        <p className="form-note">The current backend runs synchronously. Results appear as soon as its calculation completes.</p>
      </form>
    </Panel>
  )
}
