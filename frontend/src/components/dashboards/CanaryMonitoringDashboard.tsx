/**
 * Canary Deployment Monitoring Dashboard
 */

import React, { useState, useEffect } from 'react'
import {
  listDeployments,
  getDeploymentMetrics,
  triggerRollback,
  promoteDeployment,
  type CanaryDeployment,
  type CanaryMetrics,
  type ValidationTest,
} from '../../api/canary'

export default function CanaryMonitoringDashboard() {
  const [deployments, setDeployments] = useState<CanaryDeployment[]>([])
  const [selectedDeployment, setSelectedDeployment] = useState<CanaryDeployment | null>(null)
  const [metrics, setMetrics] = useState<CanaryMetrics | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(true)

  const fetchDeployments = async () => {
    try {
      const data = await listDeployments(20)
      setDeployments(data)
      setError(null)
    } catch (err) {
      setError('Failed to fetch deployments')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const fetchMetrics = async (deploymentId: string) => {
    try {
      const data = await getDeploymentMetrics(deploymentId)
      setMetrics(data)
    } catch (err) {
      console.error('Failed to fetch metrics:', err)
    }
  }

  useEffect(() => {
    fetchDeployments()
    
    if (autoRefresh) {
      const interval = setInterval(fetchDeployments, 10000) // Refresh every 10s
      return () => clearInterval(interval)
    }
  }, [autoRefresh])

  useEffect(() => {
    if (selectedDeployment) {
      fetchMetrics(selectedDeployment.deployment_id)
      
      if (autoRefresh) {
        const interval = setInterval(
          () => fetchMetrics(selectedDeployment.deployment_id),
          5000
        )
        return () => clearInterval(interval)
      }
    }
  }, [selectedDeployment, autoRefresh])

  const handleRollback = async (deploymentId: string) => {
    if (!confirm('Are you sure you want to rollback this deployment?')) return
    
    const reason = prompt('Enter rollback reason:')
    if (!reason) return

    try {
      await triggerRollback(deploymentId, reason)
      await fetchDeployments()
    } catch (err) {
      alert('Failed to trigger rollback')
      console.error(err)
    }
  }

  const handlePromote = async (deploymentId: string) => {
    if (!confirm('Are you sure you want to promote this deployment?')) return

    try {
      await promoteDeployment(deploymentId)
      await fetchDeployments()
    } catch (err) {
      alert('Failed to promote deployment')
      console.error(err)
    }
  }

  const getStatusColor = (status: string) => {
    const colors: Record<string, string> = {
      pending: '#888',
      validating: '#007bff',
      deploying: '#ff8c00',
      monitoring: '#ffc107',
      stable: '#28a745',
      failed: '#dc3545',
      rolled_back: '#6c757d',
    }
    return colors[status] || '#888'
  }

  const getTestResultColor = (result: string) => {
    const colors: Record<string, string> = {
      pass: '#28a745',
      fail: '#dc3545',
      warning: '#ffc107',
      skipped: '#888',
    }
    return colors[result] || '#888'
  }

  if (loading) {
    return <div style={{ padding: '20px' }}>Loading deployments...</div>
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
        <h1 style={{ margin: 0 }}>Canary Deployment Monitor</h1>
        <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <input
            type="checkbox"
            checked={autoRefresh}
            onChange={(e) => setAutoRefresh(e.target.checked)}
          />
          Auto-refresh
        </label>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '20px' }}>
        {/* Deployments list */}
        <div>
          <h2 style={{ fontSize: '18px', marginBottom: '10px' }}>Active Deployments</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {deployments.map((deployment) => (
              <div
                key={deployment.deployment_id}
                onClick={() => setSelectedDeployment(deployment)}
                style={{
                  padding: '15px',
                  border: '1px solid #ddd',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  backgroundColor:
                    selectedDeployment?.deployment_id === deployment.deployment_id
                      ? '#f0f8ff'
                      : 'white',
                }}
              >
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  marginBottom: '5px',
                }}>
                  <strong>{deployment.model_id}</strong>
                  <span
                    style={{
                      padding: '2px 8px',
                      borderRadius: '12px',
                      fontSize: '12px',
                      backgroundColor: getStatusColor(deployment.status),
                      color: 'white',
                    }}
                  >
                    {deployment.status}
                  </span>
                </div>
                <div style={{ fontSize: '14px', color: '#666' }}>
                  Version: {deployment.model_version}
                </div>
                <div style={{ fontSize: '14px', color: '#666' }}>
                  Mode: {deployment.mode}
                </div>
                <div style={{ fontSize: '14px', color: '#666' }}>
                  Traffic: {deployment.traffic_percentage}%
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Deployment details */}
        <div>
          {selectedDeployment ? (
            <>
              <h2 style={{ fontSize: '18px', marginBottom: '10px' }}>
                Deployment Details
              </h2>
              
              {/* Metrics */}
              {metrics && (
                <div style={{
                  padding: '15px',
                  border: '1px solid #ddd',
                  borderRadius: '8px',
                  marginBottom: '15px',
                  backgroundColor: '#f8f9fa',
                }}>
                  <h3 style={{ fontSize: '16px', marginBottom: '10px' }}>
                    Real-time Metrics
                  </h3>
                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(3, 1fr)',
                    gap: '15px',
                  }}>
                    <div>
                      <div style={{ fontSize: '12px', color: '#666' }}>
                        Requests Served
                      </div>
                      <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                        {metrics.requests_served}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '12px', color: '#666' }}>
                        Success Rate
                      </div>
                      <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                        {(metrics.success_rate * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '12px', color: '#666' }}>
                        Avg Latency
                      </div>
                      <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                        {metrics.avg_latency_ms.toFixed(0)}ms
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '12px', color: '#666' }}>
                        Error Rate
                      </div>
                      <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                        {(metrics.error_rate * 100).toFixed(2)}%
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '12px', color: '#666' }}>
                        Traffic
                      </div>
                      <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                        {metrics.current_traffic}%
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '12px', color: '#666' }}>
                        Rollbacks
                      </div>
                      <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                        {metrics.rollback_count}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Validation tests */}
              <div style={{
                padding: '15px',
                border: '1px solid #ddd',
                borderRadius: '8px',
                marginBottom: '15px',
              }}>
                <h3 style={{ fontSize: '16px', marginBottom: '10px' }}>
                  Validation Tests
                </h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  {selectedDeployment.validation_tests.map((test) => (
                    <div
                      key={test.test_id}
                      style={{
                        padding: '10px',
                        border: '1px solid #eee',
                        borderRadius: '4px',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                      }}
                    >
                      <div>
                        <div style={{ fontWeight: 'bold' }}>{test.test_name}</div>
                        <div style={{ fontSize: '12px', color: '#666' }}>
                          {test.test_type} • Score: {test.score.toFixed(3)} / Threshold: {test.threshold.toFixed(3)}
                        </div>
                      </div>
                      <span
                        style={{
                          padding: '4px 12px',
                          borderRadius: '12px',
                          fontSize: '12px',
                          backgroundColor: getTestResultColor(test.result),
                          color: 'white',
                          fontWeight: 'bold',
                        }}
                      >
                        {test.result.toUpperCase()}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Actions */}
              <div style={{ display: 'flex', gap: '10px' }}>
                {selectedDeployment.mode === 'canary' && (
                  <button
                    onClick={() => handlePromote(selectedDeployment.deployment_id)}
                    style={{
                      padding: '10px 20px',
                      backgroundColor: '#28a745',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      fontSize: '14px',
                    }}
                  >
                    Promote to Stable
                  </button>
                )}
                {!selectedDeployment.rollback_triggered && (
                  <button
                    onClick={() => handleRollback(selectedDeployment.deployment_id)}
                    style={{
                      padding: '10px 20px',
                      backgroundColor: '#dc3545',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      fontSize: '14px',
                    }}
                  >
                    Trigger Rollback
                  </button>
                )}
              </div>

              {selectedDeployment.rollback_triggered && (
                <div style={{
                  marginTop: '15px',
                  padding: '10px',
                  backgroundColor: '#fff3cd',
                  border: '1px solid #ffc107',
                  borderRadius: '4px',
                }}>
                  <strong>Rollback Triggered:</strong> {selectedDeployment.rollback_reason}
                </div>
              )}
            </>
          ) : (
            <div style={{
              height: '400px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#888',
            }}>
              Select a deployment to view details
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
