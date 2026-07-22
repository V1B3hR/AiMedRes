/**
 * Dashboard stats API module.
 */

import apiClient from './client'
import type { DashboardStats } from '../types'

export async function fetchDashboardStats(): Promise<DashboardStats> {
  const response = await apiClient.get<DashboardStats>('/api/v1/dashboard/stats')
  return response.data
}
