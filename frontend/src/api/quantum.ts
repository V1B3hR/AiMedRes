/**
 * API client for Quantum Key Management endpoints.
 */

import apiClient from './cases'

export interface CryptoKey {
  key_id: string
  key_type: 'master' | 'data_encryption' | 'session' | 'api' | 'backup'
  status: 'active' | 'rotating' | 'deprecated' | 'revoked' | 'archived'
  created_at: string
  last_rotated?: string
  expires_at?: string
  rotation_count: number
  usage_count: number
  metadata: Record<string, any>
}

export interface KeyRotationPolicy {
  enabled: boolean
  rotation_interval_days: number
  max_key_age_days: number
  grace_period_days: number
  automatic_rotation: boolean
  notify_before_rotation_days: number
  require_manual_approval: boolean
}

export interface KeyManagerStats {
  total_keys: number
  active_keys: number
  rotating_keys: number
  deprecated_keys: number
  total_rotations: number
  last_rotation?: string
  next_rotation?: string
  encryption_operations: number
  decryption_operations: number
}

export interface KeyRotationEvent {
  event_id: string
  key_id: string
  event_type: 'rotation_started' | 'rotation_completed' | 'rotation_failed'
  timestamp: string
  details: Record<string, any>
}

export async function listKeys(
  keyType?: string,
  status?: string
): Promise<CryptoKey[]> {
  const params = new URLSearchParams()
  if (keyType) params.append('key_type', keyType)
  if (status) params.append('status', status)
  
  const response = await apiClient.get(`/api/v1/quantum/keys?${params}`)
  return response.data.keys
}

export async function getKey(keyId: string): Promise<CryptoKey> {
  const response = await apiClient.get(`/api/v1/quantum/keys/${keyId}`)
  return response.data
}

export async function getKeyManagerStats(): Promise<KeyManagerStats> {
  const response = await apiClient.get('/api/v1/quantum/stats')
  return response.data
}

export async function getRotationPolicy(): Promise<KeyRotationPolicy> {
  const response = await apiClient.get('/api/v1/quantum/policy')
  return response.data
}

export async function updateRotationPolicy(
  policy: Partial<KeyRotationPolicy>
): Promise<KeyRotationPolicy> {
  const response = await apiClient.put('/api/v1/quantum/policy', policy)
  return response.data
}

export async function rotateKey(keyId: string): Promise<any> {
  const response = await apiClient.post(`/api/v1/quantum/keys/${keyId}/rotate`)
  return response.data
}

export async function getRotationHistory(
  limit: number = 50
): Promise<KeyRotationEvent[]> {
  const response = await apiClient.get('/api/v1/quantum/history', {
    params: { limit },
  })
  return response.data.events
}
