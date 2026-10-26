import React, { useState, useEffect } from 'react'
import { ResearchDisclaimerBanner } from './components/dashboards/ResearchDisclaimerBanner'
import { ClinicalProgressionExplorer } from './components/dashboards/ClinicalProgressionExplorer'
import { ModelGovernancePanel } from './components/dashboards/ModelGovernancePanel'
import BrainViewer from './components/BrainViewer'
import DicomViewer from './components/DicomViewer'
import CanaryMonitoringDashboard from './components/dashboards/CanaryMonitoringDashboard'
import QuantumKeyManagementDashboard from './components/dashboards/QuantumKeyManagementDashboard'
import { modelsApi } from './api/models'

type TabType = 'progression' | 'governance' | 'neuroimaging' | 'operations'

export default function App() {
  const [activeTab, setActiveTab] = useState<TabType>('progression')
  const [backendStatus, setBackendStatus] = useState<string>('checking...')

  useEffect(() => {
    modelsApi
      .getBackendHealth()
      .then((res) => setBackendStatus(res.status))
      .catch(() => setBackendStatus('offline_mock'))
  }, [])

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: '#0b0f19',
        color: '#f8fafc',
      }}
    >
      {/* Top Research Disclaimer Banner */}
      <ResearchDisclaimerBanner />

      {/* Main Header */}
      <header
        style={{
          borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
          background: 'rgba(17, 24, 39, 0.85)',
          backdropFilter: 'blur(16px)',
          position: 'sticky',
          top: 0,
          zIndex: 40,
        }}
      >
        <div
          style={{
            maxWidth: '1280px',
            margin: '0 auto',
            padding: '1rem 1.5rem',
            display: 'flex',
            flexWrap: 'wrap',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '1rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
            <div
              style={{
                width: '40px',
                height: '40px',
                borderRadius: '10px',
                background: 'linear-gradient(135deg, #38bdf8 0%, #6366f1 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 0 16px rgba(56, 189, 248, 0.4)',
                fontWeight: 800,
                fontSize: '1.2rem',
                color: '#ffffff',
              }}
            >
              Ai
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                <h1 style={{ fontSize: '1.25rem', fontWeight: 800, letterSpacing: '-0.02em', margin: 0 }}>
                  AiMedRes
                </h1>
                <span
                  style={{
                    fontSize: '0.7rem',
                    textTransform: 'uppercase',
                    letterSpacing: '0.08em',
                    fontWeight: 700,
                    padding: '0.15rem 0.45rem',
                    borderRadius: '4px',
                    background: 'rgba(99, 102, 241, 0.2)',
                    color: '#818cf8',
                    border: '1px solid rgba(99, 102, 241, 0.3)',
                  }}
                >
                  Research Prototype
                </span>
              </div>
              <p style={{ margin: 0, fontSize: '0.8rem', color: '#94a3b8' }}>
                Clinical Decision Support & Reproducible Progression Verification
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                background: 'rgba(15, 23, 42, 0.8)',
                padding: '0.4rem 0.85rem',
                borderRadius: '9999px',
                border: '1px solid rgba(255, 255, 255, 0.08)',
                fontSize: '0.75rem',
              }}
            >
              <span
                style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  background: backendStatus === 'healthy' ? '#10b981' : '#38bdf8',
                  boxShadow: `0 0 8px ${backendStatus === 'healthy' ? '#10b981' : '#38bdf8'}`,
                }}
              />
              <span style={{ color: '#cbd5e1' }}>
                API Status: <strong style={{ color: '#f8fafc' }}>{backendStatus}</strong>
              </span>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div
          style={{
            maxWidth: '1280px',
            margin: '0 auto',
            padding: '0 1.5rem',
            display: 'flex',
            gap: '0.5rem',
            overflowX: 'auto',
          }}
        >
          <button
            onClick={() => setActiveTab('progression')}
            style={{
              background: 'none',
              border: 'none',
              padding: '0.85rem 1.15rem',
              color: activeTab === 'progression' ? '#38bdf8' : '#94a3b8',
              fontWeight: activeTab === 'progression' ? 700 : 500,
              fontSize: '0.9rem',
              borderBottom: activeTab === 'progression' ? '2px solid #38bdf8' : '2px solid transparent',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              transition: 'all 0.2s ease',
              whiteSpace: 'nowrap',
            }}
          >
            <span>📈</span> Parkinson's Progression & What-If
          </button>

          <button
            onClick={() => setActiveTab('governance')}
            style={{
              background: 'none',
              border: 'none',
              padding: '0.85rem 1.15rem',
              color: activeTab === 'governance' ? '#38bdf8' : '#94a3b8',
              fontWeight: activeTab === 'governance' ? 700 : 500,
              fontSize: '0.9rem',
              borderBottom: activeTab === 'governance' ? '2px solid #38bdf8' : '2px solid transparent',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              transition: 'all 0.2s ease',
              whiteSpace: 'nowrap',
            }}
          >
            <span>🛡️</span> Model Card & Provenance
          </button>

          <button
            onClick={() => setActiveTab('neuroimaging')}
            style={{
              background: 'none',
              border: 'none',
              padding: '0.85rem 1.15rem',
              color: activeTab === 'neuroimaging' ? '#38bdf8' : '#94a3b8',
              fontWeight: activeTab === 'neuroimaging' ? 700 : 500,
              fontSize: '0.9rem',
              borderBottom: activeTab === 'neuroimaging' ? '2px solid #38bdf8' : '2px solid transparent',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              transition: 'all 0.2s ease',
              whiteSpace: 'nowrap',
            }}
          >
            <span>🧠</span> 3D Brain & DICOM Imaging
          </button>

          <button
            onClick={() => setActiveTab('operations')}
            style={{
              background: 'none',
              border: 'none',
              padding: '0.85rem 1.15rem',
              color: activeTab === 'operations' ? '#38bdf8' : '#94a3b8',
              fontWeight: activeTab === 'operations' ? 700 : 500,
              fontSize: '0.9rem',
              borderBottom: activeTab === 'operations' ? '2px solid #38bdf8' : '2px solid transparent',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              transition: 'all 0.2s ease',
              whiteSpace: 'nowrap',
            }}
          >
            <span>⚙️</span> Operations & Security MLOps
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <main
        style={{
          flex: 1,
          maxWidth: '1280px',
          width: '100%',
          margin: '0 auto',
          padding: '2rem 1.5rem',
        }}
      >
        {activeTab === 'progression' && <ClinicalProgressionExplorer />}

        {activeTab === 'governance' && <ModelGovernancePanel />}

        {activeTab === 'neuroimaging' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            <div
              style={{
                background: 'rgba(30, 41, 59, 0.7)',
                backdropFilter: 'blur(12px)',
                borderRadius: '12px',
                border: '1px solid rgba(255, 255, 255, 0.08)',
                padding: '1.5rem',
              }}
            >
              <h2 style={{ fontSize: '1.25rem', color: '#38bdf8', margin: '0 0 1rem 0' }}>
                🧠 Interactive 3D Neuroimaging Suite
              </h2>
              <p style={{ color: '#94a3b8', fontSize: '0.9rem', margin: '0 0 1.5rem 0' }}>
                Explore anatomical overlays and multi-planar DICOM reconstructed slices for neurodegenerative disease cohorts.
              </p>
              <BrainViewer patientId="SYNTH-PPMI-3102" mode="disease" />
            </div>

            <div
              style={{
                background: 'rgba(30, 41, 59, 0.7)',
                backdropFilter: 'blur(12px)',
                borderRadius: '12px',
                border: '1px solid rgba(255, 255, 255, 0.08)',
                padding: '1.5rem',
              }}
            >
              <h2 style={{ fontSize: '1.25rem', color: '#38bdf8', margin: '0 0 1rem 0' }}>
                🔬 DICOM Multi-Planar Series Inspector
              </h2>
              <DicomViewer patientId="SYNTH-PPMI-3102" />
            </div>
          </div>
        )}

        {activeTab === 'operations' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            <CanaryMonitoringDashboard />
            <QuantumKeyManagementDashboard />
          </div>
        )}
      </main>

      {/* Institutional Research Footer */}
      <footer
        style={{
          borderTop: '1px solid rgba(255, 255, 255, 0.08)',
          background: 'rgba(15, 23, 42, 0.95)',
          padding: '2rem 1.5rem',
          color: '#94a3b8',
          fontSize: '0.85rem',
        }}
      >
        <div
          style={{
            maxWidth: '1280px',
            margin: '0 auto',
            display: 'flex',
            flexWrap: 'wrap',
            justifyContent: 'space-between',
            alignItems: 'center',
            gap: '1rem',
          }}
        >
          <div>
            <strong style={{ color: '#f1f5f9' }}>AiMedRes Research Framework</strong> — Independent Retrospective Validation Prototype.
            <div style={{ marginTop: '0.25rem' }}>
              Prepared for UK & Poland institutional research partnerships (Parkinson's UK Project Grant 2027).
            </div>
          </div>

          <div style={{ display: 'flex', gap: '1.5rem', fontSize: '0.8rem' }}>
            <span>Licence: <strong style={{ color: '#f1f5f9' }}>MIT (Research Use)</strong></span>
            <span>Security: <strong style={{ color: '#10b981' }}>Render Advisory Remediated</strong></span>
            <span>Evidence: <strong style={{ color: '#38bdf8' }}>P1 Benchmark Frozen</strong></span>
          </div>
        </div>
      </footer>
    </div>
  )
}
