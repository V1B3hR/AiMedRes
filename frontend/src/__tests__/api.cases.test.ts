/**
 * Unit tests for the cases API module.
 *
 * Uses vi.mock to avoid real HTTP calls.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'

// Mock axios before importing the module under test
vi.mock('axios', () => {
  const mockInstance = {
    get: vi.fn(),
    post: vi.fn(),
    interceptors: {
      request: { use: vi.fn() },
      response: { use: vi.fn() },
    },
  }
  return {
    default: {
      create: () => mockInstance,
      ...mockInstance,
    },
  }
})

import axios from 'axios'
import { fetchCases, fetchCaseDetail, approveCase, login } from '../api/cases'

// Resolve the internal axios instance created by the module
const mockAxios = axios.create() as ReturnType<typeof axios.create> & {
  get: ReturnType<typeof vi.fn>
  post: ReturnType<typeof vi.fn>
}

describe('cases API', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  // ------------------------------------------------------------------
  describe('fetchCases', () => {
    it('returns parsed CaseListResponse on success', async () => {
      const payload = {
        cases: [
          {
            case_id: 'c1',
            patient_id: 'p1',
            status: 'pending',
            risk_level: 'low',
            prediction: { class: 'low', probability: 0.1 },
            created_at: '2025-01-01T00:00:00Z',
            updated_at: '2025-01-01T00:00:00Z',
            model_version: 'v1',
          },
        ],
        total: 1,
        page: 1,
        per_page: 20,
      }
      mockAxios.get.mockResolvedValueOnce({ data: payload })

      const result = await fetchCases()

      expect(mockAxios.get).toHaveBeenCalledOnce()
      expect(result.total).toBe(1)
      expect(result.cases[0].case_id).toBe('c1')
    })

    it('passes status filter as query param', async () => {
      mockAxios.get.mockResolvedValueOnce({ data: { cases: [], total: 0, page: 1, per_page: 20 } })
      await fetchCases('pending')
      const url: string = mockAxios.get.mock.calls[0][0]
      expect(url).toContain('status=pending')
    })

    it('passes pagination params', async () => {
      mockAxios.get.mockResolvedValueOnce({ data: { cases: [], total: 0, page: 2, per_page: 10 } })
      await fetchCases(undefined, 2, 10)
      const url: string = mockAxios.get.mock.calls[0][0]
      expect(url).toContain('page=2')
      expect(url).toContain('per_page=10')
    })

    it('propagates network errors', async () => {
      mockAxios.get.mockRejectedValueOnce(new Error('network error'))
      await expect(fetchCases()).rejects.toThrow('network error')
    })
  })

  // ------------------------------------------------------------------
  describe('fetchCaseDetail', () => {
    it('returns case detail for given id', async () => {
      const payload = {
        case_id: 'abc',
        patient_id: 'p2',
        status: 'completed',
        risk_level: 'high',
        prediction: { class: 'high', probability: 0.85 },
        created_at: '2025-01-01T00:00:00Z',
        updated_at: '2025-01-01T00:00:00Z',
        model_version: 'v2',
        explainability: {
          attributions: [{ feature: 'age', importance: 0.4, value: 72 }],
          uncertainty: { confidence: 0.92, total_uncertainty: 0.08 },
        },
      }
      mockAxios.get.mockResolvedValueOnce({ data: payload })

      const result = await fetchCaseDetail('abc')

      expect(mockAxios.get).toHaveBeenCalledWith('/api/v1/cases/abc')
      expect(result.explainability?.attributions[0].feature).toBe('age')
    })
  })

  // ------------------------------------------------------------------
  describe('approveCase', () => {
    it('posts correct payload for approve action', async () => {
      mockAxios.post.mockResolvedValueOnce({ data: { success: true } })

      const result = await approveCase('c1', 'approve', 'clinically appropriate', 'looks good')

      expect(mockAxios.post).toHaveBeenCalledWith('/api/v1/cases/c1/approve', {
        action: 'approve',
        rationale: 'clinically appropriate',
        notes: 'looks good',
      })
      expect(result.success).toBe(true)
    })

    it('posts correct payload for reject action without notes', async () => {
      mockAxios.post.mockResolvedValueOnce({ data: { success: true } })
      await approveCase('c2', 'reject', 'insufficient evidence')
      expect(mockAxios.post).toHaveBeenCalledWith('/api/v1/cases/c2/approve', {
        action: 'reject',
        rationale: 'insufficient evidence',
        notes: undefined,
      })
    })
  })

  // ------------------------------------------------------------------
  describe('login', () => {
    it('posts credentials to auth endpoint', async () => {
      mockAxios.post.mockResolvedValueOnce({ data: { token: 'jwt123', expires_in: 3600 } })

      const result = await login('admin', 'password')

      expect(mockAxios.post).toHaveBeenCalledWith('/api/v1/auth/login', {
        username: 'admin',
        password: 'password',
      })
      expect(result.token).toBe('jwt123')
    })

    it('throws on invalid credentials', async () => {
      mockAxios.post.mockRejectedValueOnce({ response: { status: 401 } })
      await expect(login('bad', 'creds')).rejects.toMatchObject({ response: { status: 401 } })
    })
  })
})
