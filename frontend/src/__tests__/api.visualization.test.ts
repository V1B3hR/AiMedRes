/**
 * Unit tests for the visualization API module.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'

vi.mock('axios', () => {
  const mock = {
    get: vi.fn(),
    post: vi.fn(),
    interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } },
  }
  return { default: { create: () => mock, ...mock } }
})

import axios from 'axios'
import { brainVisualizationAPI } from '../api/visualization'

const mockAxios = axios.create() as unknown as {
  get: ReturnType<typeof vi.fn>
  post: ReturnType<typeof vi.fn>
}

describe('brainVisualizationAPI', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('createOverlay sends correct request body', async () => {
    const mockResponse = {
      overlay_id: 'ov1',
      patient_id: 'p1',
      markers: [],
      regions_highlighted: ['frontal_lobe'],
      render_config: {},
    }
    mockAxios.post.mockResolvedValueOnce({ data: mockResponse })

    const result = await brainVisualizationAPI.createOverlay({
      patient_id: 'p1',
      regions_of_interest: ['frontal_lobe'],
      highlight_abnormalities: true,
    })

    expect(mockAxios.post).toHaveBeenCalledWith(
      '/brain/overlay',
      expect.objectContaining({
        patient_id: 'p1',
        regions_of_interest: ['frontal_lobe'],
        highlight_abnormalities: true,
      })
    )
    expect(result.overlay_id).toBe('ov1')
  })

  it('createDiseaseMap returns disease mapping data', async () => {
    const mockResponse = {
      map_id: 'dm1',
      patient_id: 'p1',
      disease_type: 'alzheimers',
      affected_regions: ['hippocampus'],
      severity_map: {},
    }
    mockAxios.post.mockResolvedValueOnce({ data: mockResponse })

    const result = await brainVisualizationAPI.createDiseaseMap({
      patient_id: 'p1',
      disease_type: 'alzheimers',
      severity_map: { hippocampus: 0.8 },
    })

    expect(mockAxios.post).toHaveBeenCalledWith(
      '/brain/disease-map',
      expect.objectContaining({
        patient_id: 'p1',
        disease_type: 'alzheimers',
        severity_map: { hippocampus: 0.8 },
      })
    )
    expect(result.disease_type).toBe('alzheimers')
    expect(result.affected_regions).toContain('hippocampus')
  })

  it('simulateTreatment sends treatment parameters', async () => {
    const mockResponse = {
      simulation_id: 'sim1',
      treatment_type: 'medication',
      predicted_outcome: { improvement_probability: 0.72 },
    }
    mockAxios.post.mockResolvedValueOnce({ data: mockResponse })

    const result = await brainVisualizationAPI.simulateTreatment({
      patient_id: 'p1',
      baseline_snapshot_id: 'snap-1',
      treatment_type: 'medication',
      duration_days: 30,
      efficacy_rate: 0.72,
    })

    expect(mockAxios.post).toHaveBeenCalledWith(
      '/brain/treatment-simulation',
      expect.objectContaining({ treatment_type: 'medication' })
    )
    expect(result.predicted_outcome.improvement_probability).toBeGreaterThan(0.5)
  })

  it('propagates 404 errors from the backend', async () => {
    mockAxios.post.mockRejectedValueOnce({ response: { status: 404, data: { detail: 'patient not found' } } })
    await expect(
      brainVisualizationAPI.createOverlay({ patient_id: 'unknown', regions_of_interest: [], highlight_abnormalities: false })
    ).rejects.toMatchObject({ response: { status: 404 } })
  })
})
