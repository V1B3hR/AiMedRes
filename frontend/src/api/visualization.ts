/**
 * API Client for Visualization Services
 * 
 * Provides methods to interact with P3-1 visualization endpoints:
 * - Brain visualization
 * - DICOM viewer
 */

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8080';

const api = axios.create({
  baseURL: `${API_BASE_URL}/api/v1/visualization`,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// ==================== Brain Visualization ====================

export interface BrainOverlayRequest {
  patient_id: string;
  regions_of_interest: string[];
  highlight_abnormalities?: boolean;
}

export interface DiseaseMapRequest {
  patient_id: string;
  disease_type: string;
  severity_map: Record<string, number>;
}

export interface ProgressionSnapshotRequest {
  patient_id: string;
  stage: string;
  affected_regions: Record<string, number>;
  biomarkers?: Record<string, number>;
  cognitive_scores?: Record<string, number>;
}

export interface TemporalProgressionRequest {
  patient_id: string;
  snapshot_ids: string[];
  time_scale?: string;
}

export interface TreatmentSimulationRequest {
  patient_id: string;
  baseline_snapshot_id: string;
  treatment_type: string;
  duration_days?: number;
  efficacy_rate?: number;
}

export interface TreatmentComparisonRequest {
  patient_id: string;
  simulation_ids: string[];
}

export const brainVisualizationAPI = {
  createOverlay: async (data: BrainOverlayRequest) => {
    const response = await api.post('/brain/overlay', data);
    return response.data;
  },

  createDiseaseMap: async (data: DiseaseMapRequest) => {
    const response = await api.post('/brain/disease-map', data);
    return response.data;
  },

  captureSnapshot: async (data: ProgressionSnapshotRequest) => {
    const response = await api.post('/brain/progression-snapshot', data);
    return response.data;
  },

  visualizeProgression: async (data: TemporalProgressionRequest) => {
    const response = await api.post('/brain/temporal-progression', data);
    return response.data;
  },

  simulateTreatment: async (data: TreatmentSimulationRequest) => {
    const response = await api.post('/brain/treatment-simulation', data);
    return response.data;
  },

  compareTreatments: async (data: TreatmentComparisonRequest) => {
    const response = await api.post('/brain/compare-treatments', data);
    return response.data;
  },

  getStatistics: async () => {
    const response = await api.get('/brain/statistics');
    return response.data;
  },
};

// ==================== DICOM Viewer ====================

export interface DicomSeriesListParams {
  patient_id: string;
}

export const dicomViewerAPI = {
  listSeries: async (params: DicomSeriesListParams) => {
    const response = await api.get('/dicom/series', { params });
    return response.data;
  },

  getSeriesMetadata: async (seriesId: string) => {
    const response = await api.get(`/dicom/series/${seriesId}/metadata`);
    return response.data;
  },

  getThumbnail: async (seriesId: string) => {
    const response = await api.get(`/dicom/series/${seriesId}/thumbnail`);
    return response.data;
  },

  getSlice: async (seriesId: string, sliceNumber: number, windowCenter?: number, windowWidth?: number) => {
    const params = new URLSearchParams();
    if (windowCenter !== undefined) params.append('window_center', windowCenter.toString());
    if (windowWidth !== undefined) params.append('window_width', windowWidth.toString());
    
    const response = await api.get(`/dicom/series/${seriesId}/slice/${sliceNumber}`, { params });
    return response.data;
  },

  streamSeries: (seriesId: string, onMessage: (data: any) => void) => {
    const eventSource = new EventSource(
      `${API_BASE_URL}/api/v1/visualization/dicom/series/${seriesId}/stream`
    );
    
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      onMessage(data);
    };
    
    return eventSource;
  },

  getExplainability: async (seriesId: string, data: { model_prediction: string; slice_number: number }) => {
    const response = await api.post(`/dicom/series/${seriesId}/explainability`, data);
    return response.data;
  },
};

export default api;
