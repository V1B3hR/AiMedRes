/**
 * Alerts API module for the mobile clinical companion.
 */

import apiClient from './client'
import type { AlertListResponse, ClinicalAlert } from '../types'

export async function fetchAlerts(
  acknowledged?: boolean,
  page: number = 1,
  per_page: number = 50
): Promise<AlertListResponse> {
  const params = new URLSearchParams()
  params.append('page', String(page))
  params.append('per_page', String(per_page))
  if (acknowledged !== undefined) {
    params.append('acknowledged', String(acknowledged))
  }
  const response = await apiClient.get<AlertListResponse>(
    `/api/v1/alerts?${params.toString()}`
  )
  return response.data
}

export async function acknowledgeAlert(alertId: string): Promise<ClinicalAlert> {
  const response = await apiClient.post<ClinicalAlert>(
    `/api/v1/alerts/${alertId}/acknowledge`
  )
  return response.data
}

export async function fetchUnacknowledgedAlerts(): Promise<ClinicalAlert[]> {
  const response = await fetchAlerts(false)
  return response.alerts
}
