/**
 * API client for AiMedRes backend.
 */

import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8080'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Add auth token to requests
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

export interface Case {
  case_id: string
  patient_id: string
  status: 'pending' | 'in_review' | 'completed' | 'rejected'
  risk_level: 'low' | 'moderate' | 'high' | 'critical'
  prediction: {
    class: string
    probability: number
    risk_score?: number
  }
  created_at: string
  updated_at: string
  model_version: string
}

export interface CaseDetail extends Case {
  explainability?: {
    attributions: Array<{
      feature: string
      importance: number
      value?: number
      contribution?: number
    }>
    uncertainty: {
      confidence: number
      total_uncertainty: number
      epistemic_uncertainty?: number
      aleatoric_uncertainty?: number
    }
  }
}

export interface CaseListResponse {
  cases: Case[]
  total: number
  page: number
  per_page: number
}

export async function fetchCases(
  status?: string,
  page: number = 1,
  perPage: number = 20
): Promise<CaseListResponse> {
  const params = new URLSearchParams()
  if (status) params.append('status', status)
  params.append('page', page.toString())
  params.append('per_page', perPage.toString())

  const response = await apiClient.get(`/api/v1/cases?${params}`)
  return response.data
}

export async function fetchCaseDetail(caseId: string): Promise<CaseDetail> {
  const response = await apiClient.get(`/api/v1/cases/${caseId}`)
  return response.data
}

export async function approveCase(
  caseId: string,
  action: 'approve' | 'reject' | 'request_review',
  rationale: string,
  notes?: string
): Promise<any> {
  const response = await apiClient.post(`/api/v1/cases/${caseId}/approve`, {
    action,
    rationale,
    notes,
  })
  return response.data
}

export async function login(username: string, password: string): Promise<any> {
  const response = await apiClient.post('/api/v1/auth/login', {
    username,
    password,
  })
  return response.data
}

export default apiClient
