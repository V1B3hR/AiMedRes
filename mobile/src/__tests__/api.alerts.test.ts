/**
 * Unit tests for the alerts API module.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'

vi.mock('../api/client', () => {
  const mockInstance = {
    get: vi.fn(),
    post: vi.fn(),
  }
  return { default: mockInstance }
})

import apiClient from '../api/client'
import { fetchAlerts, acknowledgeAlert, fetchUnacknowledgedAlerts } from '../api/alerts'

const mock = apiClient as unknown as {
  get: ReturnType<typeof vi.fn>
  post: ReturnType<typeof vi.fn>
}

const sampleAlert = {
  alert_id: 'a1',
  severity: 'critical' as const,
  category: 'risk_escalation' as const,
  title: 'High Risk Patient',
  message: 'Patient risk level elevated to critical.',
  case_id: 'c1',
  created_at: '2025-01-01T00:00:00Z',
  acknowledged: false,
}

describe('alerts API', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('fetchAlerts', () => {
    it('returns AlertListResponse on success', async () => {
      mock.get.mockResolvedValueOnce({
        data: { alerts: [sampleAlert], total: 1, unacknowledged_count: 1 },
      })

      const result = await fetchAlerts()

      expect(mock.get).toHaveBeenCalledOnce()
      expect(result.alerts[0].alert_id).toBe('a1')
      expect(result.unacknowledged_count).toBe(1)
    })

    it('passes acknowledged=false for unacknowledged filter', async () => {
      mock.get.mockResolvedValueOnce({ data: { alerts: [], total: 0, unacknowledged_count: 0 } })
      await fetchAlerts(false)
      const url: string = mock.get.mock.calls[0][0]
      expect(url).toContain('acknowledged=false')
    })

    it('does not append acknowledged param when undefined', async () => {
      mock.get.mockResolvedValueOnce({ data: { alerts: [], total: 0, unacknowledged_count: 0 } })
      await fetchAlerts(undefined)
      const url: string = mock.get.mock.calls[0][0]
      expect(url).not.toContain('acknowledged')
    })

    it('propagates errors', async () => {
      mock.get.mockRejectedValueOnce(new Error('timeout'))
      await expect(fetchAlerts()).rejects.toThrow('timeout')
    })
  })

  describe('acknowledgeAlert', () => {
    it('posts to correct endpoint', async () => {
      mock.post.mockResolvedValueOnce({ data: { ...sampleAlert, acknowledged: true } })

      const result = await acknowledgeAlert('a1')

      expect(mock.post).toHaveBeenCalledWith('/api/v1/alerts/a1/acknowledge')
      expect(result.acknowledged).toBe(true)
    })
  })

  describe('fetchUnacknowledgedAlerts', () => {
    it('returns only alerts list', async () => {
      mock.get.mockResolvedValueOnce({
        data: { alerts: [sampleAlert], total: 1, unacknowledged_count: 1 },
      })

      const result = await fetchUnacknowledgedAlerts()

      expect(result).toHaveLength(1)
      expect(result[0].severity).toBe('critical')
    })
  })
})
