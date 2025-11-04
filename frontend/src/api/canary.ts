/**
 * API client for Canary Deployment Monitoring endpoints.
 */

import apiClient from './cases'

export interface CanaryDeployment {
  deployment_id: string
  model_id: string
  model_version: string
  mode: 'shadow' | 'canary' | 'stable' | 'rollback'
  status: 'pending' | 'validating' | 'deploying' | 'monitoring' | 'stable' | 'failed' | 'rolled_back'
  traffic_percentage: number
  started_at: string
  completed_at?: string
  validation_tests: ValidationTest[]
  performance_metrics: Record<string, number>
  rollback_triggered: boolean
  rollback_reason?: string
}

export interface ValidationTest {
  test_id: string
  test_name: string
  test_type: 'accuracy' | 'fairness' | 'drift' | 'performance'
  result: 'pass' | 'fail' | 'warning' | 'skipped'
  score: number
  threshold: number
  passed: boolean
  details: Record<string, any>
  executed_at?: string
}

export interface CanaryMetrics {
  deployment_id: string
  current_traffic: number
  requests_served: number
  success_rate: number
  avg_latency_ms: number
  error_rate: number
  rollback_count: number
}

export async function listDeployments(
  limit: number = 20
): Promise<CanaryDeployment[]> {
  const response = await apiClient.get('/api/v1/canary/deployments', {
    params: { limit },
  })
  return response.data.deployments
}

export async function getDeployment(
  deploymentId: string
): Promise<CanaryDeployment> {
  const response = await apiClient.get(`/api/v1/canary/deployments/${deploymentId}`)
  return response.data
}

export async function getDeploymentMetrics(
  deploymentId: string
): Promise<CanaryMetrics> {
  const response = await apiClient.get(
    `/api/v1/canary/deployments/${deploymentId}/metrics`
  )
  return response.data
}

export async function triggerRollback(
  deploymentId: string,
  reason: string
): Promise<any> {
  const response = await apiClient.post(
    `/api/v1/canary/deployments/${deploymentId}/rollback`,
    { reason }
  )
  return response.data
}

export async function promoteDeployment(
  deploymentId: string
): Promise<any> {
  const response = await apiClient.post(
    `/api/v1/canary/deployments/${deploymentId}/promote`
  )
  return response.data
}
