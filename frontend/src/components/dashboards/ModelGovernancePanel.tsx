import React, { useEffect, useState } from 'react'
import {
  DEFAULT_BENCHMARK_METRICS,
  ModelCard,
  modelsApi,
} from '../../api/models'

export const ModelGovernancePanel: React.FC = () => {
  const [modelCard, setModelCard] = useState<ModelCard | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    modelsApi
      .getModelCard('latest')
      .then((data) => setModelCard(data))
      .catch(() => setModelCard(null))
      .finally(() => setIsLoading(false))
  }, [])

  const metrics = DEFAULT_BENCHMARK_METRICS

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Top Banner: Verification Status */}
      <div
        style={{
          background: 'rgba(30, 41, 59, 0.7)',
          backdropFilter: 'blur(12px)',
          borderRadius: '12px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          padding: '1.25rem 1.5rem',
          display: 'flex',
          flexWrap: 'wrap',
          gap: '1rem',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <div>
          <div style={{ fontSize: '0.8rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Registered Benchmark Baseline
          </div>
          <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#f8fafc' }}>
            {modelCard?.name || metrics.study_name}
          </div>
        </div>

        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          <span
            style={{
              padding: '0.35rem 0.75rem',
              borderRadius: '6px',
              fontSize: '0.8rem',
              fontWeight: 600,
              background: 'rgba(56, 189, 248, 0.15)',
              color: '#38bdf8',
              border: '1px solid rgba(56, 189, 248, 0.3)',
            }}
          >
            Version: {modelCard?.version || 'v1.1.0'}
          </span>
          <span
            style={{
              padding: '0.35rem 0.75rem',
              borderRadius: '6px',
              fontSize: '0.8rem',
              fontWeight: 600,
              background: 'rgba(234, 179, 8, 0.15)',
              color: '#eab308',
              border: '1px solid rgba(234, 179, 8, 0.3)',
            }}
          >
            Status: {modelCard?.status || 'inference_disabled'} (Research Fixture)
          </span>
        </div>
      </div>

      {/* Model Benchmark Performance Table */}
      <div
        style={{
          background: 'rgba(30, 41, 59, 0.6)',
          backdropFilter: 'blur(12px)',
          borderRadius: '12px',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          padding: '1.5rem',
        }}
      >
        <h3
          style={{
            margin: '0 0 0.5rem 0',
            fontSize: '1.1rem',
            fontWeight: 600,
            color: '#38bdf8',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
          }}
        >
          <span>🏆</span> 5-Fold Patient-Isolated Cross-Validation & Bootstrap Metrics
        </h3>
        <p style={{ margin: '0 0 1.25rem 0', fontSize: '0.85rem', color: '#94a3b8' }}>
          Frozen execution baseline on synthetic PPMI fixture (StratifiedGroupKFold on PATNO, 1000 bootstrap resamples, alpha=0.05).
        </p>

        <div style={{ overflowX: 'auto' }}>
          <table
            style={{
              width: '100%',
              borderCollapse: 'collapse',
              fontSize: '0.9rem',
              textAlign: 'left',
            }}
          >
            <thead>
              <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.1)', color: '#94a3b8' }}>
                <th style={{ padding: '0.75rem 1rem' }}>Model Architecture</th>
                <th style={{ padding: '0.75rem 1rem' }}>ROC-AUC</th>
                <th style={{ padding: '0.75rem 1rem' }}>95% Bootstrap CI</th>
                <th style={{ padding: '0.75rem 1rem' }}>Brier Calibration Score</th>
                <th style={{ padding: '0.75rem 1rem' }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {metrics.models.map((m, idx) => (
                <tr
                  key={m.name}
                  style={{
                    borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
                    background: idx === 0 ? 'rgba(56, 189, 248, 0.05)' : 'transparent',
                  }}
                >
                  <td style={{ padding: '0.85rem 1rem', fontWeight: 600, color: '#f1f5f9' }}>
                    {m.name} {idx === 0 && <span style={{ color: '#38bdf8', fontSize: '0.75rem' }}>(Baseline)</span>}
                  </td>
                  <td style={{ padding: '0.85rem 1rem', fontWeight: 700, color: '#38bdf8' }}>
                    {m.roc_auc.toFixed(3)}
                  </td>
                  <td style={{ padding: '0.85rem 1rem', color: '#cbd5e1' }}>
                    {m.ci_95_lower.toFixed(3)} – {m.ci_95_upper.toFixed(3)}
                  </td>
                  <td style={{ padding: '0.85rem 1rem', color: '#cbd5e1' }}>
                    {m.brier_score.toFixed(3)}
                  </td>
                  <td style={{ padding: '0.85rem 1rem' }}>
                    <span
                      style={{
                        padding: '0.2rem 0.5rem',
                        borderRadius: '4px',
                        fontSize: '0.75rem',
                        background: 'rgba(16, 185, 129, 0.15)',
                        color: '#10b981',
                        border: '1px solid rgba(16, 185, 129, 0.3)',
                      }}
                    >
                      Reproduced
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Cryptographic Provenance Manifest */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
          gap: '1.5rem',
        }}
      >
        <div
          style={{
            background: 'rgba(30, 41, 59, 0.6)',
            backdropFilter: 'blur(12px)',
            borderRadius: '12px',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            padding: '1.5rem',
          }}
        >
          <h3
            style={{
              margin: '0 0 1rem 0',
              fontSize: '1.1rem',
              fontWeight: 600,
              color: '#38bdf8',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
            }}
          >
            <span>🔐</span> Cryptographic Audit & Provenance
          </h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            <div style={{ background: '#0f172a', padding: '0.75rem', borderRadius: '8px' }}>
              <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Study Config SHA-256</div>
              <div style={{ fontSize: '0.75rem', fontFamily: 'monospace', color: '#38bdf8', wordBreak: 'break-all' }}>
                {metrics.provenance.config_hash}
              </div>
            </div>

            <div style={{ background: '#0f172a', padding: '0.75rem', borderRadius: '8px' }}>
              <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Synthetic Fixture SHA-256</div>
              <div style={{ fontSize: '0.75rem', fontFamily: 'monospace', color: '#38bdf8', wordBreak: 'break-all' }}>
                {metrics.provenance.fixture_hash}
              </div>
            </div>

            <div style={{ background: '#0f172a', padding: '0.75rem', borderRadius: '8px' }}>
              <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Verified Git Commit</div>
              <div style={{ fontSize: '0.85rem', fontFamily: 'monospace', color: '#f1f5f9' }}>
                {metrics.provenance.git_head}
              </div>
            </div>
          </div>
        </div>

        {/* Intended Use & Regulatory Boundaries */}
        <div
          style={{
            background: 'rgba(30, 41, 59, 0.6)',
            backdropFilter: 'blur(12px)',
            borderRadius: '12px',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            padding: '1.5rem',
          }}
        >
          <h3
            style={{
              margin: '0 0 1rem 0',
              fontSize: '1.1rem',
              fontWeight: 600,
              color: '#f59e0b',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
            }}
          >
            <span>⚖️</span> Intended Use & Boundaries (Model Card)
          </h3>

          <div style={{ fontSize: '0.85rem', color: '#cbd5e1', lineHeight: 1.6 }}>
            <p style={{ margin: '0 0 0.75rem 0' }}>
              <strong>Intended Use:</strong> {modelCard?.intended_use || 'Local reproducibility verification on a non-identifiable synthetic fixture; no patient-level inference.'}
            </p>

            <div style={{ fontWeight: 600, color: '#f8fafc', marginBottom: '0.4rem' }}>
              Explicit Boundaries & Limitations:
            </div>
            <ul style={{ margin: '0', paddingLeft: '1.25rem', color: '#94a3b8' }}>
              {(modelCard?.limitations || [
                'Research use only',
                'No API-served artifact is approved or mounted',
                'Not a clinical model or patient-level prediction service',
              ]).map((lim, idx) => (
                <li key={idx} style={{ marginBottom: '0.25rem' }}>
                  {lim}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
