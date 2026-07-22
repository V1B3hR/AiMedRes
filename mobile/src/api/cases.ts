/**
 * Cases API module for the mobile clinical companion.
 */

import apiClient from './client'
import type {
  AuthToken,
  Case,
  CaseApprovalResponse,
  CaseDetail,
  CaseListResponse,
  CaseStatus,
  LoginRequest,
} from '../types'

// ─── Auth ──────────────────────────────────────────────────────────────────

export async function login(username: string, password: string): Promise<AuthToken> {
  const payload: LoginRequest = { username, password }
  const response = await apiClient.post<AuthToken>('/api/v1/auth/login', payload)
  return response.data
}

// ─── Cases ─────────────────────────────────────────────────────────────────

export async function fetchCases(
  status?: CaseStatus,
  page: number = 1,
  per_page: number = 20
): Promise<CaseListResponse> {
  const params = new URLSearchParams()
  params.append('page', String(page))
  params.append('per_page', String(per_page))
  if (status) params.append('status', status)

  const response = await apiClient.get<CaseListResponse>(
    `/api/v1/cases?${params.toString()}`
  )
  return response.data
}

export async function fetchCaseDetail(caseId: string): Promise<CaseDetail> {
  const response = await apiClient.get<CaseDetail>(`/api/v1/cases/${caseId}`)
  return response.data
}

export async function approveCase(
  caseId: string,
  action: 'approve' | 'reject',
  rationale: string,
  notes?: string
): Promise<CaseApprovalResponse> {
  const response = await apiClient.post<CaseApprovalResponse>(
    `/api/v1/cases/${caseId}/approve`,
    { action, rationale, notes }
  )
  return response.data
}

// ─── High-risk cases convenience helper ────────────────────────────────────

export async function fetchHighRiskCases(): Promise<Case[]> {
  const response = await fetchCases('pending')
  return response.cases.filter(
    (c) => c.risk_level === 'high' || c.risk_level === 'critical'
  )
}
