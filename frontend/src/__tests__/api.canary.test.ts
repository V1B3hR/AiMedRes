/**
 * Unit tests for the canary deployment API module.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'

vi.mock('./cases', () => {
  const mock = {
    get: vi.fn(),
    post: vi.fn(),
    interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } },
  }
  return { default: mock }
})

import apiClient from '../api/cases'
import {
  listDeployments,
  getDeployment,
  getDeploymentMetrics,
  triggerRollback,
  type CanaryDeployment,
  type CanaryMetrics,
} from '../api/canary'

const mockClient = apiClient as unknown as { get: ReturnType<typeof vi.fn>; post: ReturnType<typeof vi.fn> }

const sampleDeployment: CanaryDeployment = {
  deployment_id: 'd1',
  model_id: 'model-alzheimer-v3',
  model_version: '3.0.1',
  mode: 'canary',
  status: 'monitoring',
  traffic_percentage: 10,
  started_at: '2025-06-01T00:00:00Z',
  validation_tests: [],
  performance_metrics: { accuracy: 0.94, latency_ms: 82 },
  rollback_triggered: false,
}

const sampleMetrics: CanaryMetrics = {
  deployment_id: 'd1',
  current_traffic: 10,
  requests_served: 500,
  success_rate: 0.998,
  avg_latency_ms: 82,
  error_rate: 0.002,
  rollback_count: 0,
}

describe('canary API', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('listDeployments', () => {
    it('returns array of deployments', async () => {
      mockClient.get.mockResolvedValueOnce({ data: { deployments: [sampleDeployment] } })

      const result = await listDeployments(10)

      expect(mockClient.get).toHaveBeenCalledWith('/api/v1/canary/deployments', { params: { limit: 10 } })
      expect(result).toHaveLength(1)
      expect(result[0].deployment_id).toBe('d1')
    })

    it('defaults limit to 20', async () => {
      mockClient.get.mockResolvedValueOnce({ data: { deployments: [] } })
      await listDeployments()
      expect(mockClient.get).toHaveBeenCalledWith('/api/v1/canary/deployments', { params: { limit: 20 } })
    })
  })

  describe('getDeployment', () => {
    it('fetches a single deployment by id', async () => {
      mockClient.get.mockResolvedValueOnce({ data: sampleDeployment })

      const result = await getDeployment('d1')

      expect(mockClient.get).toHaveBeenCalledWith('/api/v1/canary/deployments/d1')
      expect(result.model_version).toBe('3.0.1')
      expect(result.traffic_percentage).toBe(10)
    })
  })

  describe('getDeploymentMetrics', () => {
    it('returns metrics for a deployment', async () => {
      mockClient.get.mockResolvedValueOnce({ data: sampleMetrics })

      const result = await getDeploymentMetrics('d1')

      expect(mockClient.get).toHaveBeenCalledWith('/api/v1/canary/deployments/d1/metrics')
      expect(result.success_rate).toBeCloseTo(0.998)
      expect(result.error_rate).toBeLessThan(0.01)
    })
  })

  describe('triggerRollback', () => {
    it('posts rollback with reason', async () => {
      mockClient.post.mockResolvedValueOnce({ data: { success: true, rolled_back_at: '2025-06-01T01:00:00Z' } })

      const result = await triggerRollback('d1', 'accuracy degradation')

      expect(mockClient.post).toHaveBeenCalledWith(
        '/api/v1/canary/deployments/d1/rollback',
        { reason: 'accuracy degradation' }
      )
      expect(result.success).toBe(true)
    })

    it('propagates server errors', async () => {
      mockClient.post.mockRejectedValueOnce({ response: { status: 409, data: { detail: 'already rolled back' } } })
      await expect(triggerRollback('d1', 'test')).rejects.toMatchObject({
        response: { status: 409 },
      })
    })
  })
})
