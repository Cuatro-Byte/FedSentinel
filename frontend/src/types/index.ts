export type RunStatus = 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED'
export type ThreatLevel = 'SAFE' | 'SUSPICIOUS' | 'MALICIOUS'
export type ImpactLevel = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
export type ResponseAction = 'ACCEPT' | 'DOWN_WEIGHT' | 'QUARANTINE'
export type RecoveryStatus = 'NOT_REQUIRED' | 'TRIGGERED' | 'COMPLETED' | 'FAILED'

export interface SimulationSummary {
  run_id: string
  scenario: string
  status: RunStatus
  current_round: number
  round_count: number
  created_at: string | null
}

export interface SimulationRun extends SimulationSummary {
  client_count: number
  model_version: string
  attack_enabled: boolean
  seed: number
  schema_version: string
  detector_version: string
  impact_version: string
  recovery_version: string
  completed_at: string | null
}

export interface SimulationRequest {
  client_count: number
  rounds: number
  scenario: string
  attack_enabled: boolean
  attacker_count: number
  intensity: number | null
  start_round: number
  targets: string[] | null
  seed: number
  background: boolean
}

export interface SimulationCreated {
  run_id: string
  status: RunStatus
}

export interface FLRound {
  round_id: number
  run_id?: string
  status: string
  model_version: string | null
  update_count: number
  accepted_count: number
  suspicious_count?: number
  quarantined_count: number
  downweighted_count?: number
  recovery_triggered: boolean
  created_at?: string | null
}

export interface ClientState {
  client_id: string
  reputation_score: number
  latest_threat_score: number | null
  latest_impact_score: number | null
  action: ResponseAction | null
  rounds_participated: number
  suspicious_count: number
  malicious_count: number
  quarantine_count: number
}

export interface ThreatResult {
  update_id: string
  client_id: string
  round_id: number
  threat_score: number
  threat_level: ThreatLevel
  action: ResponseAction
  anomaly_score: number
  similarity_score: number
  reputation_score: number
  feature_summary: Record<string, number>
  explanation_codes: string[]
  detector_version: string
  created_at: string | null
}

export interface ImpactResult {
  update_id: string
  client_id: string
  round_id: number
  impact_score: number
  influence_estimate: number | null
  parameter_displacement: number | null
  aggregation_weight: number | null
  estimated_accuracy_change: number | null
  estimated_loss_change: number | null
  impact_level: ImpactLevel
  explanation_codes: string[]
  impact_breakdown?: Record<string, number>
  top_impacted_layers?: Array<Record<string, unknown>>
  impact_version: string
  created_at: string | null
}

export interface MetricRecord {
  run_id: string
  round_id: number
  model_version: string
  accuracy: number
  loss: number
  attack_success_rate: number | null
  malicious_updates: number
  suspicious_updates: number
  quarantined_updates: number
  accepted_updates: number
  downweighted_updates: number
  recovery_triggered: boolean
  recovery_count: number
  created_at: string | null
}

export interface RecoveryRecord {
  recovery_id: string
  run_id: string
  round_id: number
  trigger: string
  affected_update_ids: string[]
  excluded_client_ids: string[]
  previous_model_version: string | null
  recovered_model_version: string | null
  before_accuracy: number | null
  after_accuracy: number | null
  before_loss: number | null
  after_loss: number | null
  recovery_status: RecoveryStatus
  recovery_version: string
  selected_action?: string | null
  details?: Record<string, unknown>
  created_at: string | null
}

export interface AuditEvent {
  id: number
  run_id: string
  round_id: number | null
  client_id: string | null
  update_id: string | null
  event_type: string
  severity: 'INFO' | 'WARN' | 'ALERT' | string
  message: string
  model_version: string | null
  detector_version: string | null
  timestamp: string | null
}

export interface DashboardData {
  run: SimulationRun | null
  rounds: FLRound[]
  clients: ClientState[]
  threats: ThreatResult[]
  impacts: ImpactResult[]
  metrics: MetricRecord[]
  recoveries: RecoveryRecord[]
  audit: AuditEvent[]
}

export interface ClientAnalytics extends ClientState {
  threat: ThreatResult | null
  impact: ImpactResult | null
}
