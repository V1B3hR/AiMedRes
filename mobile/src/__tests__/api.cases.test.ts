/**
 * Unit tests for the cases API module.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'

// Mock the API client
vi.mock('../api/client', () => {
  const mockInstance = {
    get: vi.fn(),
    post: vi.fn(),
  }
  return { default: mockInstance }
})

import apiClient from '../api/client'
import { fetchCases, fetchCaseDetail, approveCase, fetchHighRiskCases, login } from '../api/cases'

const mock = apiClient as unknown as {
  get: ReturnType<typeof vi.fn>
  post: ReturnType<typeof vi.fn>
}

describe('cases API', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('fetchCases', () => {
    it('returns CaseListResponse on success', async () => {
      const payload = {
        cases: [
          {
            case_id: 'c1',
            patient_id: 'p1',
            status: 'pending',
            risk_level: 'high',
            prediction: { class: 'high', probability: 0.85 },
            created_at: '2025-01-01T00:00:00Z',
            updated_at: '2025-01-01T00:00:00Z',
            model_version: 'v2',
          },
        ],
        total: 1,
        page: 1,
        per_page: 20,
      }
      mock.get.mockResolvedValueOnce({ data: payload })

      const result = await fetchCases()

      expect(mock.get).toHaveBeenCalledOnce()
      expect(result.total).toBe(1)
      expect(result.cases[0].case_id).toBe('c1')
    })

    it('appends status filter to query string', async () => {
      mock.get.mockResolvedValueOnce({ data: { cases: [], total: 0, page: 1, per_page: 20 } })
      await fetchCases('pending')
      const url: string = mock.get.mock.calls[0][0]
      expect(url).toContain('status=pending')
    })

    it('appends pagination params', async () => {
      mock.get.mockResolvedValueOnce({ data: { cases: [], total: 0, page: 2, per_page: 10 } })
      await fetchCases(undefined, 2, 10)
      const url: string = mock.get.mock.calls[0][0]
      expect(url).toContain('page=2')
      expect(url).toContain('per_page=10')
    })

    it('propagates network errors', async () => {
      mock.get.mockRejectedValueOnce(new Error('network error'))
      await expect(fetchCases()).rejects.toThrow('network error')
    })
  })

  describe('fetchCaseDetail', () => {
    it('returns full case detail', async () => {
      const payload = {
        case_id: 'abc',
        patient_id: 'p2',
        status: 'in_review',
        risk_level: 'critical',
        prediction: { class: 'critical', probability: 0.92 },
        created_at: '2025-01-01T00:00:00Z',
        updated_at: '2025-01-01T00:00:00Z',
        model_version: 'v3',
        explainability: {
          attributions: [{ feature: 'age', importance: 0.4, value: 75 }],
          uncertainty: { confidence: 0.88, total_uncertainty: 0.12 },
        },
      }
      mock.get.mockResolvedValueOnce({ data: payload })

      const result = await fetchCaseDetail('abc')

      expect(mock.get).toHaveBeenCalledWith('/api/v1/cases/abc')
      expect(result.explainability?.attributions[0].feature).toBe('age')
      expect(result.risk_level).toBe('critical')
    })
  })

  describe('approveCase', () => {
    it('posts approve payload correctly', async () => {
      mock.post.mockResolvedValueOnce({ data: { success: true, case_id: 'c1', action: 'approve' } })

      const result = await approveCase('c1', 'approve', 'clinically indicated', 'looks good')

      expect(mock.post).toHaveBeenCalledWith('/api/v1/cases/c1/approve', {
        action: 'approve',
        rationale: 'clinically indicated',
        notes: 'looks good',
      })
      expect(result.success).toBe(true)
    })

    it('posts reject payload without optional notes', async () => {
      mock.post.mockResolvedValueOnce({ data: { success: true, case_id: 'c2', action: 'reject' } })
      await approveCase('c2', 'reject', 'insufficient evidence')
      expect(mock.post).toHaveBeenCalledWith('/api/v1/cases/c2/approve', {
        action: 'reject',
        rationale: 'insufficient evidence',
        notes: undefined,
      })
    })
  })

  describe('fetchHighRiskCases', () => {
    it('filters to only high and critical risk cases', async () => {
      mock.get.mockResolvedValueOnce({
        data: {
          cases: [
            { case_id: 'a', risk_level: 'low', status: 'pending', patient_id: 'p1', prediction: { class: 'low', probability: 0.1 }, created_at: '', updated_at: '', model_version: 'v1' },
            { case_id: 'b', risk_level: 'high', status: 'pending', patient_id: 'p2', prediction: { class: 'high', probability: 0.8 }, created_at: '', updated_at: '', model_version: 'v1' },
            { case_id: 'c', risk_level: 'critical', status: 'pending', patient_id: 'p3', prediction: { class: 'critical', probability: 0.95 }, created_at: '', updated_at: '', model_version: 'v1' },
            { case_id: 'd', risk_level: 'moderate', status: 'pending', patient_id: 'p4', prediction: { class: 'moderate', probability: 0.5 }, created_at: '', updated_at: '', model_version: 'v1' },
          ],
          total: 4,
          page: 1,
          per_page: 20,
        },
      })

      const result = await fetchHighRiskCases()

      expect(result).toHaveLength(2)
      expect(result.map((c) => c.case_id)).toEqual(['b', 'c'])
    })
  })

  describe('login', () => {
    it('posts credentials and returns token', async () => {
      mock.post.mockResolvedValueOnce({ data: { token: 'jwt-abc', expires_in: 3600 } })

      const result = await login('admin', 'pass')

      expect(mock.post).toHaveBeenCalledWith('/api/v1/auth/login', {
        username: 'admin',
        password: 'pass',
      })
      expect(result.token).toBe('jwt-abc')
    })
  })
})
