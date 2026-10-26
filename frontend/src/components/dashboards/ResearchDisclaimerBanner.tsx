import React from 'react'

export const ResearchDisclaimerBanner: React.FC = () => {
  return (
    <aside
      aria-label="Research Use Only Disclaimer"
      style={{
        background: 'linear-gradient(90deg, #1e1b4b 0%, #311042 50%, #1e1b4b 100%)',
        borderBottom: '1px solid rgba(244, 63, 94, 0.4)',
        padding: '0.75rem 1.5rem',
        color: '#f8fafc',
        boxShadow: '0 2px 12px rgba(0, 0, 0, 0.4)',
      }}
    >
      <div
        style={{
          maxWidth: '1280px',
          margin: '0 auto',
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '1rem',
          fontSize: '0.85rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <span
            style={{
              padding: '0.2rem 0.6rem',
              borderRadius: '4px',
              fontWeight: 800,
              fontSize: '0.75rem',
              letterSpacing: '0.05em',
              background: '#f43f5e',
              color: '#ffffff',
              boxShadow: '0 0 10px rgba(244, 63, 94, 0.6)',
            }}
          >
            RESEARCH USE ONLY
          </span>
          <span style={{ color: '#e2e8f0' }}>
            Not for clinical diagnosis, treatment decisions, or patient care. Operates on synthetic PPMI fixtures.
          </span>
        </div>

        <div style={{ display: 'flex', gap: '1rem', fontSize: '0.8rem' }}>
          <span style={{ color: '#94a3b8' }}>
            Audit Gate: <strong style={{ color: '#38bdf8' }}>P1/P2 Research Baseline</strong>
          </span>
          <span style={{ color: '#94a3b8' }}>
            Data: <strong style={{ color: '#38bdf8' }}>Synthetic Fixture (60 pts)</strong>
          </span>
        </div>
      </div>
    </aside>
  )
}
