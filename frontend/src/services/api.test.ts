import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from './api'

afterEach(() => vi.unstubAllGlobals())

describe('FedSentinel API client', () => {
  it('reads simulation runs from the configured backend', async () => {
    const payload = [{ run_id: 'RUN-001', scenario: 'normal', status: 'COMPLETED', current_round: 2, round_count: 2, created_at: null }]
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(payload), { status: 200, headers: { 'Content-Type': 'application/json' } }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(api.listSimulations()).resolves.toEqual(payload)
    expect(fetchMock).toHaveBeenCalledWith('/api/v1/simulations', expect.objectContaining({ headers: expect.objectContaining({ 'Content-Type': 'application/json' }) }))
  })

  it('preserves structured backend errors', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      detail: { error: { code: 'INVALID_CONFIGURATION', message: 'Unsupported scenario' } },
    }), { status: 400, headers: { 'Content-Type': 'application/json' } })))

    const request = api.createSimulation({ client_count: 5, rounds: 2, scenario: 'invalid', attack_enabled: true })
    await expect(request).rejects.toMatchObject({ status: 400, code: 'INVALID_CONFIGURATION', message: 'Unsupported scenario' })
  })

  it('loads every analytics endpoint for a dashboard run', async () => {
    const fetchMock = vi.fn((input: string | URL | Request) => {
      const url = String(input)
      const body = url.endsWith('/simulations/RUN-002')
        ? { run_id: 'RUN-002', scenario: 'normal', status: 'COMPLETED', current_round: 1, round_count: 1, client_count: 1, model_version: 'm1', attack_enabled: false, seed: 42, schema_version: 'schema-v1', detector_version: 'v1', impact_version: 'v1', recovery_version: 'v1', created_at: null, completed_at: null }
        : []
      return Promise.resolve(new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } }))
    })
    vi.stubGlobal('fetch', fetchMock)

    const result = await api.getDashboard('RUN-002')
    expect(result.run?.run_id).toBe('RUN-002')
    expect(fetchMock).toHaveBeenCalledTimes(9)
  })

  it('reads validation records from the validation endpoint', async () => {
    const valRecords = [
      {
        run_id: 'RUN-003',
        round_id: 1,
        model_version: 'model_v1',
        validation_loss: 0.25,
        validation_accuracy: 0.95,
        loss_spiked: false,
        baseline_loss: 0.25,
        baseline_accuracy: 0.95,
        loss_delta: 0.0,
        accuracy_delta: 0.0,
        validation_status: 'HEALTHY',
        created_at: null,
      },
    ]
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(valRecords), { status: 200, headers: { 'Content-Type': 'application/json' } }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(api.getValidation('RUN-003')).resolves.toEqual(valRecords)
    expect(fetchMock).toHaveBeenCalledWith('/api/v1/validation/RUN-003', expect.objectContaining({ headers: expect.objectContaining({ 'Content-Type': 'application/json' }) }))
  })
})
