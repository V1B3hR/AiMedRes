/**
 * API client for DICOM/3D Viewer endpoints.
 */

import apiClient from './cases'

export interface BrainRegion {
  region_id: string
  name: string
  coordinates: number[]
  volume: number
  affected: boolean
  severity?: number
}

export interface ViewerSession {
  session_id: string
  viewer_type: 'dicom' | 'brain_3d'
  created_at: string
  status: string
}

export interface BrainVisualization {
  regions: BrainRegion[]
  disease_stage: string
  atrophy_map: any
  explainability?: {
    affected_regions: string[]
    confidence: number
  }
}

export async function createViewerSession(
  type: 'dicom' | 'brain_3d',
  studyId?: string
): Promise<ViewerSession> {
  const response = await apiClient.post('/api/viewer/session', {
    viewer_type: type,
    study_id: studyId,
  })
  return response.data
}

export async function getBrainVisualization(
  patientId: string,
  diseaseStage?: string
): Promise<BrainVisualization> {
  const params = new URLSearchParams()
  if (diseaseStage) params.append('stage', diseaseStage)
  
  const response = await apiClient.get(
    `/api/viewer/brain/${patientId}?${params}`
  )
  return response.data
}

export async function getDICOMStudy(studyId: string): Promise<any> {
  const response = await apiClient.get(`/api/viewer/dicom/study/${studyId}`)
  return response.data
}

export async function streamDICOMSeries(
  studyId: string,
  seriesId: string
): Promise<any> {
  const response = await apiClient.get(
    `/api/viewer/dicom/series/${studyId}/${seriesId}/stream`
  )
  return response.data
}
