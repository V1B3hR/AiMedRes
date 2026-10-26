/**
 * API client for Model Registry and Research Reproducibility services.
 */

import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8080'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 5000,
})

// Optional authorization interceptor
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

export interface ModelSummary {
  model_id: string
  name: string
  version: string
  status: string
}

export interface ModelCard {
  name: string
  version: string
  type: string
  intended_use: string
  validation_metrics?: Record<string, any> | null
  dataset_provenance: string
  limitations: string[]
  last_updated: string
  status: string
}

export interface SyntheticPatientProfile {
  patno: string
  age: number
  gender: 'Female' | 'Male'
  disease_duration_months: number
  baseline_updrs_iii: number
  putamen_sbr_left: number
  putamen_sbr_right: number
  caudate_sbr_left: number
  caudate_sbr_right: number
  mmse_score: number
  motor_progression_risk: number
  risk_level: 'Low' | 'Moderate' | 'High'
  confidence_interval_95: [number, number]
  shap_attributions: Array<{
    feature: string
    display_name: string
    value: number | string
    contribution: number // positive increases progression risk, negative decreases
    unit?: string
  }>
}

export interface BenchmarkMetrics {
  study_name: string
  dataset_name: string
  cross_validation: string
  folds: number
  n_bootstraps: number
  models: Array<{
    name: string
    roc_auc: number
    ci_95_lower: number
    ci_95_upper: number
    brier_score: number
  }>
  provenance: {
    config_hash: string
    fixture_hash: string
    git_head: string
    execution_time_seconds: number
  }
}

// Verified evidence baseline from evidence/ppmi_synthetic_fixture/2026-10-25/EVIDENCE_PACK.md
export const DEFAULT_BENCHMARK_METRICS: BenchmarkMetrics = {
  study_name: 'ppmi-parkinsons-synthetic-reproducibility',
  dataset_name: 'synthetic_pet_sbr_fixture.csv (170 visits, 60 patients)',
  cross_validation: '5-fold StratifiedGroupKFold (patient-isolated by PATNO)',
  folds: 5,
  n_bootstraps: 1000,
  models: [
    {
      name: 'Logistic Regression (Calibrated)',
      roc_auc: 0.907,
      ci_95_lower: 0.859,
      ci_95_upper: 0.948,
      brier_score: 0.108,
    },
    {
      name: 'Random Forest Classifier',
      roc_auc: 0.843,
      ci_95_lower: 0.776,
      ci_95_upper: 0.903,
      brier_score: 0.142,
    },
    {
      name: 'Support Vector Machine (RBF)',
      roc_auc: 0.835,
      ci_95_lower: 0.753,
      ci_95_upper: 0.904,
      brier_score: 0.132,
    },
  ],
  provenance: {
    config_hash: '9e303ec951d88632793aa12c8deb81d9eb627775ed7aba60de12b3062a40fe01',
    fixture_hash: 'e803e33abbaea34197e5506e5b73111d0766edc54bc898883926ccb2e44687ae',
    git_head: 'cc1487bd19dce0892d4f8d594c0f3363d69f32ba',
    execution_time_seconds: 17.89,
  },
}

// Curated synthetic patient profiles from the synthetic fixture for interactive exploration
export const SYNTHETIC_PATIENT_PROFILES: SyntheticPatientProfile[] = [
  {
    patno: 'SYNTH-PPMI-3102',
    age: 63,
    gender: 'Male',
    disease_duration_months: 24,
    baseline_updrs_iii: 28.5,
    putamen_sbr_left: 0.78,
    putamen_sbr_right: 0.92,
    caudate_sbr_left: 1.45,
    caudate_sbr_right: 1.62,
    mmse_score: 29,
    motor_progression_risk: 0.76,
    risk_level: 'High',
    confidence_interval_95: [0.68, 0.84],
    shap_attributions: [
      { feature: 'putamen_sbr_left', display_name: 'Putamen SBR (Left)', value: 0.78, contribution: 0.28, unit: 'ratio' },
      { feature: 'baseline_updrs_iii', display_name: 'Baseline UPDRS Part III', value: 28.5, contribution: 0.22, unit: 'pts' },
      { feature: 'disease_duration_months', display_name: 'Disease Duration', value: 24, contribution: 0.14, unit: 'mo' },
      { feature: 'caudate_sbr_left', display_name: 'Caudate SBR (Left)', value: 1.45, contribution: 0.08, unit: 'ratio' },
      { feature: 'age', display_name: 'Patient Age', value: 63, contribution: 0.04, unit: 'yr' },
    ],
  },
  {
    patno: 'SYNTH-PPMI-4055',
    age: 58,
    gender: 'Female',
    disease_duration_months: 12,
    baseline_updrs_iii: 16.0,
    putamen_sbr_left: 1.35,
    putamen_sbr_right: 1.42,
    caudate_sbr_left: 2.10,
    caudate_sbr_right: 2.05,
    mmse_score: 30,
    motor_progression_risk: 0.28,
    risk_level: 'Low',
    confidence_interval_95: [0.21, 0.36],
    shap_attributions: [
      { feature: 'putamen_sbr_left', display_name: 'Putamen SBR (Left)', value: 1.35, contribution: -0.24, unit: 'ratio' },
      { feature: 'baseline_updrs_iii', display_name: 'Baseline UPDRS Part III', value: 16.0, contribution: -0.18, unit: 'pts' },
      { feature: 'caudate_sbr_right', display_name: 'Caudate SBR (Right)', value: 2.05, contribution: -0.12, unit: 'ratio' },
      { feature: 'disease_duration_months', display_name: 'Disease Duration', value: 12, contribution: -0.06, unit: 'mo' },
      { feature: 'age', display_name: 'Patient Age', value: 58, contribution: -0.02, unit: 'yr' },
    ],
  },
  {
    patno: 'SYNTH-PPMI-2287',
    age: 71,
    gender: 'Male',
    disease_duration_months: 18,
    baseline_updrs_iii: 22.0,
    putamen_sbr_left: 1.05,
    putamen_sbr_right: 1.12,
    caudate_sbr_left: 1.75,
    caudate_sbr_right: 1.80,
    mmse_score: 27,
    motor_progression_risk: 0.52,
    risk_level: 'Moderate',
    confidence_interval_95: [0.44, 0.61],
    shap_attributions: [
      { feature: 'age', display_name: 'Patient Age', value: 71, contribution: 0.16, unit: 'yr' },
      { feature: 'putamen_sbr_left', display_name: 'Putamen SBR (Left)', value: 1.05, contribution: 0.12, unit: 'ratio' },
      { feature: 'baseline_updrs_iii', display_name: 'Baseline UPDRS Part III', value: 22.0, contribution: 0.05, unit: 'pts' },
      { feature: 'mmse_score', display_name: 'Cognitive Score (MMSE)', value: 27, contribution: 0.04, unit: 'pts' },
      { feature: 'caudate_sbr_left', display_name: 'Caudate SBR (Left)', value: 1.75, contribution: -0.07, unit: 'ratio' },
    ],
  },
]

export const modelsApi = {
  async getModelList(): Promise<ModelSummary[]> {
    try {
      const response = await apiClient.get<ModelSummary[]>('/api/v1/model/')
      return response.data
    } catch {
      return [
        {
          model_id: 'ppmi_synthetic_fixture_v1',
          name: 'Synthetic PPMI-schema reproducibility benchmark',
          version: 'v1.1.0',
          status: 'inference_disabled',
        },
      ]
    }
  },

  async getModelCard(version: string = 'latest'): Promise<ModelCard> {
    try {
      const response = await apiClient.get<ModelCard>(`/api/v1/model/${version}`)
      return response.data
    } catch {
      return {
        name: 'Synthetic PPMI-schema reproducibility benchmark',
        version: 'v1.1.0',
        type: 'research_fixture',
        intended_use:
          'Local reproducibility verification on a non-identifiable synthetic fixture; no patient-level inference.',
        validation_metrics: null,
        dataset_provenance: 'Synthetic fixture only; no controlled PPMI result is registered.',
        limitations: [
          'Research use only',
          'No API-served artifact is approved or mounted',
          'Not a clinical model or patient-level prediction service',
        ],
        last_updated: '2026-10-25',
        status: 'inference_disabled',
      }
    }
  },

  async getBackendHealth(): Promise<{ status: string; timestamp: string }> {
    try {
      const response = await apiClient.get<{ status: string; timestamp: string }>('/health')
      return response.data
    } catch {
      return { status: 'mock_demo_mode', timestamp: new Date().toISOString() }
    }
  },
}
