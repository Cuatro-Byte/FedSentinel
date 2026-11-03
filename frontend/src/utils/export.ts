import type { DashboardData } from '../types'

const download = (contents: BlobPart, filename: string, type: string) => {
  const url = URL.createObjectURL(new Blob([contents], { type }))
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(url)
}

const escapeCsv = (value: unknown) => {
  const raw = Array.isArray(value) || (value && typeof value === 'object')
    ? JSON.stringify(value)
    : String(value ?? '')
  return '"' + raw.replaceAll('"', '""') + '"'
}

const asCsv = (rows: unknown[][]) =>
  rows.map((row) => row.map(escapeCsv).join(',')).join('\n')

export const exportJson = (data: DashboardData, runId: string) =>
  download(JSON.stringify(data, null, 2), 'fedsentinel-' + runId + '.json', 'application/json')

export const exportRecoveryCsv = (data: DashboardData, runId: string) => {
  const headers = ['recovery_id', 'run_id', 'round_id', 'status', 'trigger', 'affected_update_ids', 'excluded_client_ids', 'previous_model_version', 'recovered_model_version', 'before_accuracy', 'after_accuracy', 'before_loss', 'after_loss', 'created_at']
  const rows = data.recoveries.map((record) => [
    record.recovery_id, record.run_id, record.round_id, record.recovery_status,
    record.trigger, record.affected_update_ids, record.excluded_client_ids,
    record.previous_model_version, record.recovered_model_version,
    record.before_accuracy, record.after_accuracy, record.before_loss,
    record.after_loss, record.created_at,
  ])
  download(asCsv([headers, ...rows]), 'fedsentinel-' + runId + '-recovery.csv', 'text/csv;charset=utf-8')
}

export const exportAnalyticsCsv = (data: DashboardData, runId: string) => {
  const impactByUpdate = new Map(data.impacts.map((impact) => [impact.update_id, impact]))
  const headers = ['run_id', 'round_id', 'client_id', 'update_id', 'threat_score', 'threat_level', 'action', 'anomaly_score', 'similarity_score', 'reputation_score', 'impact_score', 'impact_level', 'influence_estimate', 'explanation_codes']
  const rows = data.threats.map((threat) => {
    const impact = impactByUpdate.get(threat.update_id)
    return [
      runId, threat.round_id, threat.client_id, threat.update_id, threat.threat_score,
      threat.threat_level, threat.action, threat.anomaly_score, threat.similarity_score,
      threat.reputation_score, impact?.impact_score, impact?.impact_level,
      impact?.influence_estimate, threat.explanation_codes,
    ]
  })
  download(asCsv([headers, ...rows]), 'fedsentinel-' + runId + '-analytics.csv', 'text/csv;charset=utf-8')
}
