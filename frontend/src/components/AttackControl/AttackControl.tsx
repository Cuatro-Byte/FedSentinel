import { useMemo, useState } from 'react'
import { Activity, Bug, FlaskConical, Play, Shield, ShieldAlert, TimerReset, Zap } from 'lucide-react'
import type { SimulationRequest } from '../../types'
import { Panel } from '../shared/Ui'

const scenarios = [
  { value: 'normal', label: 'Normal', description: 'Clean federation baseline', icon: Shield },
  { value: 'model_poisoning', label: 'Model poisoning', description: 'Manipulated parameter updates', icon: Zap },
  { value: 'label_poisoning', label: 'Label poisoning', description: 'Corrupted local labels', icon: FlaskConical },
  { value: 'backdoor', label: 'Backdoor', description: 'Triggered data-plane attack', icon: Bug },
  { value: 'mixed_attack', label: 'Mixed attack', description: 'Multiple attack techniques', icon: ShieldAlert },
  { value: 'sleeper', label: 'Sleeper', description: 'Delayed adversarial behavior', icon: TimerReset },
]

const initial: SimulationRequest = {
  client_count: 10,
  rounds: 5,
  scenario: 'backdoor',
  attack_enabled: true,
  attacker_count: 1,
  intensity: 0.8,
  start_round: 1,
  targets: null,
  seed: 42,
  background: false,
}

export function AttackControl({
  onStart,
  running,
  expanded = false,
}: {
  onStart: (request: SimulationRequest) => Promise<void>
  running: boolean
  expanded?: boolean
}) {
  const [request, setRequest] = useState<SimulationRequest>(initial)
  const [targetText, setTargetText] = useState('')
  const update = <K extends keyof SimulationRequest>(key: K, value: SimulationRequest[K]) =>
    setRequest((current) => ({ ...current, [key]: value }))

  const errors = useMemo(() => {
    const next: string[] = []
    if (request.client_count < 1 || request.client_count > 100) next.push('Clients must be between 1 and 100.')
    if (request.rounds < 1 || request.rounds > 100) next.push('Rounds must be between 1 and 100.')
    if ((request.attacker_count ?? 0) > request.client_count) next.push('Attackers cannot exceed clients.')
    if ((request.start_round ?? 1) > request.rounds) next.push('Attack start must be within the configured rounds.')
    return next
  }, [request])

  const submit = () => {
    if (errors.length) return
    const targets = targetText.split(',').map((item) => item.trim()).filter(Boolean)
    void onStart({ ...request, targets: targets.length ? targets : null })
  }

  return (
    <Panel
      title="Scenario builder"
      eyebrow="Simulation control"
      className={`attack-control ${expanded ? 'expanded-builder' : ''}`}
      action={<Activity aria-hidden="true" />}
    >
      <form onSubmit={(event) => { event.preventDefault(); submit() }}>
        <fieldset>
          <legend>Attack scenario</legend>
          <div className="scenario-grid">
            {scenarios.map((scenario) => (
              <button
                type="button"
                key={scenario.value}
                className={request.scenario === scenario.value ? 'scenario-option selected' : 'scenario-option'}
                onClick={() => update('scenario', scenario.value)}
              >
                <scenario.icon />
                <span>
                  <strong>{scenario.label}</strong>
                  <small>{scenario.description}</small>
                </span>
              </button>
            ))}
          </div>
        </fieldset>
        <div className="builder-sections">
          <fieldset>
            <legend>Federation</legend>
            <div className="form-grid">
              <label>
                <span>Clients</span>
                <input
                  type="number"
                  min="1"
                  max="100"
                  value={request.client_count}
                  onChange={(e) => update('client_count', Number(e.target.value))}
                />
              </label>
              <label>
                <span>Rounds</span>
                <input
                  type="number"
                  min="1"
                  max="100"
                  value={request.rounds}
                  onChange={(e) => update('rounds', Number(e.target.value))}
                />
              </label>
              <label>
                <span>Random seed</span>
                <input
                  type="number"
                  value={request.seed ?? 42}
                  onChange={(e) => update('seed', Number(e.target.value))}
                />
              </label>
            </div>
          </fieldset>
          <fieldset disabled={!request.attack_enabled}>
            <legend>Attack configuration</legend>
            <div className="form-grid">
              <label>
                <span>Attackers</span>
                <input
                  type="number"
                  min="0"
                  max={request.client_count}
                  value={request.attacker_count ?? 1}
                  onChange={(e) => update('attacker_count', Number(e.target.value))}
                />
              </label>
              <label>
                <span>Intensity</span>
                <input
                  type="number"
                  min="0"
                  step="0.1"
                  value={request.intensity ?? ''}
                  onChange={(e) => update('intensity', e.target.value === '' ? null : Number(e.target.value))}
                />
              </label>
              <label>
                <span>Start round</span>
                <input
                  type="number"
                  min="1"
                  max={request.rounds}
                  value={request.start_round ?? 1}
                  onChange={(e) => update('start_round', Number(e.target.value))}
                />
              </label>
              {expanded && (
                <label className="span-2">
                  <span>Target client IDs <small>Optional, comma-separated</small></span>
                  <input
                    value={targetText}
                    onChange={(e) => setTargetText(e.target.value)}
                    placeholder="client-7, client-9"
                  />
                </label>
              )}
            </div>
          </fieldset>
        </div>
        <label className="switch-row">
          <span>
            <strong>Attack simulation</strong>
            <small>Attack ground truth remains isolated from Sentinel inference.</small>
          </span>
          <input
            type="checkbox"
            checked={request.attack_enabled}
            onChange={(e) => update('attack_enabled', e.target.checked)}
          />
        </label>
        <label className="switch-row">
          <span>
            <strong>Background execution</strong>
            <small>Return immediately and follow real run status through live refresh.</small>
          </span>
          <input
            type="checkbox"
            checked={request.background ?? false}
            onChange={(e) => update('background', e.target.checked)}
          />
        </label>
        {!!errors.length && (
          <div className="form-errors" role="alert">
            {errors.map((error) => (
              <span key={error}>{error}</span>
            ))}
          </div>
        )}
        <button
          className="primary-button launch-button"
          type="submit"
          disabled={running || !!errors.length}
        >
          <Play />
          {running ? 'Starting simulation…' : 'Launch simulation'}
        </button>
      </form>
    </Panel>
  )
}
