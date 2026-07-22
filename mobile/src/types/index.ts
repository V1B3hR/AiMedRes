/**
 * Shared types for the AiMedRes Mobile Clinical Companion App.
 * These mirror the web frontend types to maintain API compatibility.
 */

// ─── Case Types ────────────────────────────────────────────────────────────

export type CaseStatus = 'pending' | 'in_review' | 'completed' | 'rejected'

export type RiskLevel = 'low' | 'moderate' | 'high' | 'critical'

export interface CasePrediction {
  class: string
  probability: number
  risk_score?: number
}

export interface Case {
  case_id: string
  patient_id: string
  status: CaseStatus
  risk_level: RiskLevel
  prediction: CasePrediction
  created_at: string
  updated_at: string
  model_version: string
}

export interface FeatureAttribution {
  feature: string
  importance: number
  value?: number
  contribution?: number
}

export interface Uncertainty {
  confidence: number
  total_uncertainty: number
  epistemic_uncertainty?: number
  aleatoric_uncertainty?: number
}

export interface Explainability {
  attributions: FeatureAttribution[]
  uncertainty: Uncertainty
}

export interface CaseDetail extends Case {
  explainability?: Explainability
  clinical_notes?: string
  patient_age?: number
  patient_gender?: string
  condition_type?: string
}

export interface CaseListResponse {
  cases: Case[]
  total: number
  page: number
  per_page: number
}

export interface CaseApprovalResponse {
  success: boolean
  case_id: string
  action: string
  timestamp?: string
}

// ─── Alert Types ───────────────────────────────────────────────────────────

export type AlertSeverity = 'info' | 'warning' | 'critical'

export type AlertCategory =
  | 'risk_escalation'
  | 'model_drift'
  | 'system_health'
  | 'compliance'
  | 'patient_status'

export interface ClinicalAlert {
  alert_id: string
  severity: AlertSeverity
  category: AlertCategory
  title: string
  message: string
  case_id?: string
  patient_id?: string
  created_at: string
  acknowledged: boolean
  acknowledged_at?: string
}

export interface AlertListResponse {
  alerts: ClinicalAlert[]
  total: number
  unacknowledged_count: number
}

// ─── Dashboard Types ───────────────────────────────────────────────────────

export interface DashboardStats {
  pending_cases: number
  in_review_cases: number
  completed_today: number
  high_risk_cases: number
  critical_alerts: number
  model_accuracy: number
  avg_processing_time_ms: number
}

// ─── Auth Types ────────────────────────────────────────────────────────────

export interface LoginRequest {
  username: string
  password: string
}

export interface AuthToken {
  token: string
  expires_in: number
  user_id?: string
  role?: string
}

// ─── Navigation Types ──────────────────────────────────────────────────────

export type RootTabParamList = {
  Dashboard: undefined
  Cases: undefined
  Alerts: undefined
}

export type CasesStackParamList = {
  CasesList: undefined
  CaseDetail: { caseId: string }
}
