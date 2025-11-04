/**
 * Quantum Key Management Dashboard
 */

import React, { useState, useEffect } from 'react'
import {
  listKeys,
  getKeyManagerStats,
  getRotationPolicy,
  updateRotationPolicy,
  rotateKey,
  getRotationHistory,
  type CryptoKey,
  type KeyManagerStats,
  type KeyRotationPolicy,
  type KeyRotationEvent,
} from '../../api/quantum'

export default function QuantumKeyManagementDashboard() {
  const [keys, setKeys] = useState<CryptoKey[]>([])
  const [stats, setStats] = useState<KeyManagerStats | null>(null)
  const [policy, setPolicy] = useState<KeyRotationPolicy | null>(null)
  const [history, setHistory] = useState<KeyRotationEvent[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedTab, setSelectedTab] = useState<'keys' | 'policy' | 'history'>('keys')
  const [autoRefresh, setAutoRefresh] = useState(true)

  const fetchData = async () => {
    try {
      const [keysData, statsData, policyData, historyData] = await Promise.all([
        listKeys(),
        getKeyManagerStats(),
        getRotationPolicy(),
        getRotationHistory(50),
      ])
      setKeys(keysData)
      setStats(statsData)
      setPolicy(policyData)
      setHistory(historyData)
      setError(null)
    } catch (err) {
      setError('Failed to fetch key management data')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
    
    if (autoRefresh) {
      const interval = setInterval(fetchData, 30000) // Refresh every 30s
      return () => clearInterval(interval)
    }
  }, [autoRefresh])

  const handleRotateKey = async (keyId: string) => {
    if (!confirm('Are you sure you want to rotate this key?')) return

    try {
      await rotateKey(keyId)
      await fetchData()
    } catch (err) {
      alert('Failed to rotate key')
      console.error(err)
    }
  }

  const handleUpdatePolicy = async (updates: Partial<KeyRotationPolicy>) => {
    if (!policy) return

    try {
      const updated = await updateRotationPolicy(updates)
      setPolicy(updated)
    } catch (err) {
      alert('Failed to update rotation policy')
      console.error(err)
    }
  }

  const getStatusColor = (status: string) => {
    const colors: Record<string, string> = {
      active: '#28a745',
      rotating: '#ff8c00',
      deprecated: '#ffc107',
      revoked: '#dc3545',
      archived: '#6c757d',
    }
    return colors[status] || '#888'
  }

  const getKeyTypeIcon = (type: string) => {
    const icons: Record<string, string> = {
      master: '🔑',
      data_encryption: '🔒',
      session: '⏱️',
      api: '🔌',
      backup: '💾',
    }
    return icons[type] || '🔐'
  }

  const formatDate = (dateString?: string) => {
    if (!dateString) return 'N/A'
    return new Date(dateString).toLocaleString()
  }

  if (loading) {
    return <div style={{ padding: '20px' }}>Loading key management data...</div>
  }

  if (error) {
    return <div style={{ padding: '20px', color: 'red' }}>{error}</div>
  }

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '20px',
      }}>
        <h1 style={{ margin: 0 }}>Quantum Key Management</h1>
        <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <input
            type="checkbox"
            checked={autoRefresh}
            onChange={(e) => setAutoRefresh(e.target.checked)}
          />
          Auto-refresh
        </label>
      </div>

      {/* Statistics Cards */}
      {stats && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(4, 1fr)',
          gap: '15px',
          marginBottom: '20px',
        }}>
          <div style={{
            padding: '20px',
            border: '1px solid #ddd',
            borderRadius: '8px',
            backgroundColor: '#f8f9fa',
          }}>
            <div style={{ fontSize: '12px', color: '#666', marginBottom: '5px' }}>
              Total Keys
            </div>
            <div style={{ fontSize: '32px', fontWeight: 'bold' }}>
              {stats.total_keys}
            </div>
          </div>
          <div style={{
            padding: '20px',
            border: '1px solid #ddd',
            borderRadius: '8px',
            backgroundColor: '#f8f9fa',
          }}>
            <div style={{ fontSize: '12px', color: '#666', marginBottom: '5px' }}>
              Active Keys
            </div>
            <div style={{ fontSize: '32px', fontWeight: 'bold', color: '#28a745' }}>
              {stats.active_keys}
            </div>
          </div>
          <div style={{
            padding: '20px',
            border: '1px solid #ddd',
            borderRadius: '8px',
            backgroundColor: '#f8f9fa',
          }}>
            <div style={{ fontSize: '12px', color: '#666', marginBottom: '5px' }}>
              Total Rotations
            </div>
            <div style={{ fontSize: '32px', fontWeight: 'bold' }}>
              {stats.total_rotations}
            </div>
          </div>
          <div style={{
            padding: '20px',
            border: '1px solid #ddd',
            borderRadius: '8px',
            backgroundColor: '#f8f9fa',
          }}>
            <div style={{ fontSize: '12px', color: '#666', marginBottom: '5px' }}>
              Encryption Ops
            </div>
            <div style={{ fontSize: '32px', fontWeight: 'bold' }}>
              {stats.encryption_operations}
            </div>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div style={{
        display: 'flex',
        gap: '10px',
        marginBottom: '20px',
        borderBottom: '2px solid #ddd',
      }}>
        {(['keys', 'policy', 'history'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setSelectedTab(tab)}
            style={{
              padding: '10px 20px',
              border: 'none',
              backgroundColor: 'transparent',
              cursor: 'pointer',
              fontSize: '16px',
              fontWeight: selectedTab === tab ? 'bold' : 'normal',
              borderBottom: selectedTab === tab ? '3px solid #007bff' : 'none',
              marginBottom: '-2px',
            }}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Keys Tab */}
      {selectedTab === 'keys' && (
        <div>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(350px, 1fr))',
            gap: '15px',
          }}>
            {keys.map((key) => (
              <div
                key={key.key_id}
                style={{
                  padding: '15px',
                  border: '1px solid #ddd',
                  borderRadius: '8px',
                  backgroundColor: 'white',
                }}
              >
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: '10px',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontSize: '24px' }}>
                      {getKeyTypeIcon(key.key_type)}
                    </span>
                    <strong>{key.key_type.replace('_', ' ').toUpperCase()}</strong>
                  </div>
                  <span
                    style={{
                      padding: '4px 10px',
                      borderRadius: '12px',
                      fontSize: '12px',
                      backgroundColor: getStatusColor(key.status),
                      color: 'white',
                      fontWeight: 'bold',
                    }}
                  >
                    {key.status.toUpperCase()}
                  </span>
                </div>
                
                <div style={{ fontSize: '12px', color: '#666', marginBottom: '5px' }}>
                  ID: {key.key_id.substring(0, 16)}...
                </div>
                <div style={{ fontSize: '12px', color: '#666', marginBottom: '5px' }}>
                  Created: {formatDate(key.created_at)}
                </div>
                <div style={{ fontSize: '12px', color: '#666', marginBottom: '5px' }}>
                  Last Rotated: {formatDate(key.last_rotated)}
                </div>
                <div style={{ fontSize: '12px', color: '#666', marginBottom: '10px' }}>
                  Rotations: {key.rotation_count} • Usage: {key.usage_count}
                </div>
                
                {key.status === 'active' && (
                  <button
                    onClick={() => handleRotateKey(key.key_id)}
                    style={{
                      width: '100%',
                      padding: '8px',
                      backgroundColor: '#007bff',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      fontSize: '12px',
                    }}
                  >
                    Rotate Key
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Policy Tab */}
      {selectedTab === 'policy' && policy && (
        <div style={{
          maxWidth: '600px',
          margin: '0 auto',
        }}>
          <div style={{
            padding: '20px',
            border: '1px solid #ddd',
            borderRadius: '8px',
            backgroundColor: 'white',
          }}>
            <h3 style={{ marginTop: 0 }}>Rotation Policy Configuration</h3>
            
            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <input
                  type="checkbox"
                  checked={policy.enabled}
                  onChange={(e) => handleUpdatePolicy({ enabled: e.target.checked })}
                />
                <strong>Enable Automatic Key Rotation</strong>
              </label>
            </div>

            <div style={{ marginBottom: '15px' }}>
              <label style={{ display: 'block', marginBottom: '5px' }}>
                Rotation Interval (days)
              </label>
              <input
                type="number"
                value={policy.rotation_interval_days}
                onChange={(e) => handleUpdatePolicy({ rotation_interval_days: Number(e.target.value) })}
                style={{
                  width: '100%',
                  padding: '8px',
                  border: '1px solid #ddd',
                  borderRadius: '4px',
                }}
              />
            </div>

            <div style={{ marginBottom: '15px' }}>
              <label style={{ display: 'block', marginBottom: '5px' }}>
                Maximum Key Age (days)
              </label>
              <input
                type="number"
                value={policy.max_key_age_days}
                onChange={(e) => handleUpdatePolicy({ max_key_age_days: Number(e.target.value) })}
                style={{
                  width: '100%',
                  padding: '8px',
                  border: '1px solid #ddd',
                  borderRadius: '4px',
                }}
              />
            </div>

            <div style={{ marginBottom: '15px' }}>
              <label style={{ display: 'block', marginBottom: '5px' }}>
                Grace Period (days)
              </label>
              <input
                type="number"
                value={policy.grace_period_days}
                onChange={(e) => handleUpdatePolicy({ grace_period_days: Number(e.target.value) })}
                style={{
                  width: '100%',
                  padding: '8px',
                  border: '1px solid #ddd',
                  borderRadius: '4px',
                }}
              />
            </div>

            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <input
                  type="checkbox"
                  checked={policy.require_manual_approval}
                  onChange={(e) => handleUpdatePolicy({ require_manual_approval: e.target.checked })}
                />
                Require Manual Approval for Rotation
              </label>
            </div>

            <div style={{
              padding: '15px',
              backgroundColor: '#e3f2fd',
              borderRadius: '4px',
              fontSize: '14px',
            }}>
              <strong>🔒 Quantum-Safe Encryption:</strong> All keys use hybrid Kyber768/AES-256 
              encryption for post-quantum security.
            </div>
          </div>
        </div>
      )}

      {/* History Tab */}
      {selectedTab === 'history' && (
        <div>
          <div style={{
            border: '1px solid #ddd',
            borderRadius: '8px',
            overflow: 'hidden',
          }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ backgroundColor: '#f8f9fa' }}>
                  <th style={{ padding: '12px', textAlign: 'left', borderBottom: '1px solid #ddd' }}>
                    Timestamp
                  </th>
                  <th style={{ padding: '12px', textAlign: 'left', borderBottom: '1px solid #ddd' }}>
                    Key ID
                  </th>
                  <th style={{ padding: '12px', textAlign: 'left', borderBottom: '1px solid #ddd' }}>
                    Event Type
                  </th>
                  <th style={{ padding: '12px', textAlign: 'left', borderBottom: '1px solid #ddd' }}>
                    Details
                  </th>
                </tr>
              </thead>
              <tbody>
                {history.map((event) => (
                  <tr key={event.event_id} style={{ borderBottom: '1px solid #eee' }}>
                    <td style={{ padding: '12px' }}>
                      {formatDate(event.timestamp)}
                    </td>
                    <td style={{ padding: '12px', fontFamily: 'monospace', fontSize: '12px' }}>
                      {event.key_id.substring(0, 16)}...
                    </td>
                    <td style={{ padding: '12px' }}>
                      <span
                        style={{
                          padding: '4px 8px',
                          borderRadius: '4px',
                          fontSize: '12px',
                          backgroundColor:
                            event.event_type === 'rotation_completed'
                              ? '#d4edda'
                              : event.event_type === 'rotation_failed'
                              ? '#f8d7da'
                              : '#fff3cd',
                          color:
                            event.event_type === 'rotation_completed'
                              ? '#155724'
                              : event.event_type === 'rotation_failed'
                              ? '#721c24'
                              : '#856404',
                        }}
                      >
                        {event.event_type.replace('_', ' ').toUpperCase()}
                      </span>
                    </td>
                    <td style={{ padding: '12px', fontSize: '12px', color: '#666' }}>
                      {JSON.stringify(event.details).substring(0, 50)}...
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
