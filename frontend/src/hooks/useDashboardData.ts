import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../services/api'
import type { DashboardData } from '../types'

const emptyData: DashboardData = {
  run: null,
  rounds: [],
  clients: [],
  threats: [],
  impacts: [],
  metrics: [],
  recoveries: [],
  audit: [],
}

export function useDashboardData(runId: string | null, autoRefresh: boolean) {
  const [data, setData] = useState<DashboardData>(emptyData)
  const [loading, setLoading] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const requestId = useRef(0)

  const refresh = useCallback(async (quiet = false) => {
    if (!runId) {
      setData(emptyData)
      return
    }
    const currentRequest = ++requestId.current
    quiet ? setRefreshing(true) : setLoading(true)
    try {
      const next = await api.getDashboard(runId)
      if (currentRequest !== requestId.current) return
      setData(next)
      setError(null)
      setLastUpdated(new Date())
    } catch (caught) {
      if (currentRequest !== requestId.current) return
      setError(caught instanceof Error ? caught.message : 'Unable to load dashboard data')
    } finally {
      if (currentRequest === requestId.current) {
        setLoading(false)
        setRefreshing(false)
      }
    }
  }, [runId])

  useEffect(() => {
    void refresh(false)
  }, [refresh])

  useEffect(() => {
    if (!autoRefresh || !runId) return
    const timer = window.setInterval(() => void refresh(true), 3000)
    return () => window.clearInterval(timer)
  }, [autoRefresh, refresh, runId])

  return { data, loading, refreshing, error, lastUpdated, refresh }
}
