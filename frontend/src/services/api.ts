import type {
  AuditEvent,
  ClientState,
  DashboardData,
  FLRound,
  ImpactResult,
  MetricRecord,
  RecoveryRecord,
  SimulationCreated,
  SimulationRequest,
  SimulationRun,
  SimulationSummary,
  ThreatResult,
  ValidationRecord,
} from '../types'

const configuredBase = import.meta.env.VITE_API_BASE_URL as string | undefined
export const API_BASE_URL = (configuredBase || '/api/v1').replace(/\/$/, '')

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public code?: string,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...init?.headers,
    },
  })

  if (!response.ok) {
    const body = await response.json().catch(() => null)
    const detail = body?.detail?.error ?? body?.error
    throw new ApiError(
      detail?.message || `Request failed with status ${response.status}`,
      response.status,
      detail?.code,
    )
  }

  return response.json() as Promise<T>
}

async function optionalList<T>(path: string): Promise<T[]> {
  try {
    return await request<T[]>(path)
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) return []
    throw error
  }
}

export const api = {
  health: () => request<{ status: string; service: string; api_version: string }>('/health'),
  listSimulations: () => request<SimulationSummary[]>('/simulations'),
  createSimulation: (payload: SimulationRequest) =>
    request<SimulationCreated>('/simulations', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  getSimulation: (runId: string) => request<SimulationRun>(`/simulations/${runId}`),
  getRounds: (runId: string) => optionalList<FLRound>(`/rounds/${runId}`),
  getClients: (runId: string) => optionalList<ClientState>(`/clients/${runId}`),
  getThreats: (runId: string) => optionalList<ThreatResult>(`/threats/${runId}`),
  getImpacts: (runId: string) => optionalList<ImpactResult>(`/impact/${runId}`),
  getMetrics: (runId: string) => optionalList<MetricRecord>(`/metrics/${runId}`),
  getRecoveries: (runId: string) => optionalList<RecoveryRecord>(`/recovery/${runId}`),
  getAudit: (runId: string) => optionalList<AuditEvent>(`/audit/${runId}`),
  getValidation: (runId: string) => optionalList<ValidationRecord>(`/validation/${runId}`),
  getDashboard: async (runId: string): Promise<DashboardData> => {
    const [run, rounds, clients, threats, impacts, metrics, recoveries, audit, validation] =
      await Promise.all([
        api.getSimulation(runId),
        api.getRounds(runId),
        api.getClients(runId),
        api.getThreats(runId),
        api.getImpacts(runId),
        api.getMetrics(runId),
        api.getRecoveries(runId),
        api.getAudit(runId),
        api.getValidation(runId),
      ])
    return { run, rounds, clients, threats, impacts, metrics, recoveries, audit, validation }
  },
}
