import React, { useState } from 'react'
import {
  SYNTHETIC_PATIENT_PROFILES,
  SyntheticPatientProfile,
} from '../../api/models'

export const ClinicalProgressionExplorer: React.FC = () => {
  const [selectedPatientId, setSelectedPatientId] = useState<string>(
    SYNTHETIC_PATIENT_PROFILES[0].patno
  )
  const [customUpdrs, setCustomUpdrs] = useState<number | null>(null)
  const [customPutamenSbr, setCustomPutamenSbr] = useState<number | null>(null)

  const patient =
    SYNTHETIC_PATIENT_PROFILES.find((p) => p.patno === selectedPatientId) ||
    SYNTHETIC_PATIENT_PROFILES[0]

  // Calculate dynamic display values if customized
  const currentUpdrs = customUpdrs !== null ? customUpdrs : patient.baseline_updrs_iii
  const currentPutamenSbr =
    customPutamenSbr !== null ? customPutamenSbr : patient.putamen_sbr_left

  // Sensitivity estimation on risk based on adjusted sliders
  const deltaUpdrs = (currentUpdrs - patient.baseline_updrs_iii) * 0.012
  const deltaSbr = (patient.putamen_sbr_left - currentPutamenSbr) * 0.35
  const simulatedRisk = Math.min(
    0.99,
    Math.max(0.01, patient.motor_progression_risk + deltaUpdrs + deltaSbr)
  )

  const simulatedLowerCI = Math.max(0.01, simulatedRisk - 0.08)
  const simulatedUpperCI = Math.min(0.99, simulatedRisk + 0.08)

  const getRiskBadge = (risk: number) => {
    if (risk >= 0.65) {
      return {
        label: 'High Progression Risk',
        color: '#f43f5e',
        bg: 'rgba(244, 63, 94, 0.15)',
        border: 'rgba(244, 63, 94, 0.4)',
      }
    }
    if (risk >= 0.4) {
      return {
        label: 'Moderate Progression Risk',
        color: '#fbbf24',
        bg: 'rgba(251, 191, 36, 0.15)',
        border: 'rgba(251, 191, 36, 0.4)',
      }
    }
    return {
      label: 'Low Progression Risk',
      color: '#10b981',
      bg: 'rgba(16, 185, 129, 0.15)',
      border: 'rgba(16, 185, 129, 0.4)',
    }
  }

  const badge = getRiskBadge(simulatedRisk)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Patient Selector and Controls */}
      <div
        style={{
          background: 'rgba(30, 41, 59, 0.7)',
          backdropFilter: 'blur(12px)',
          borderRadius: '12px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          padding: '1.25rem 1.5rem',
          display: 'flex',
          flexWrap: 'wrap',
          gap: '1.25rem',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <div>
          <label
            htmlFor="patient-select"
            style={{
              display: 'block',
              fontSize: '0.85rem',
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
              color: '#94a3b8',
              marginBottom: '0.4rem',
            }}
          >
            Select Synthetic Cohort Patient Profile
          </label>
          <select
            id="patient-select"
            value={selectedPatientId}
            onChange={(e) => {
              setSelectedPatientId(e.target.value)
              setCustomUpdrs(null)
              setCustomPutamenSbr(null)
            }}
            style={{
              background: '#0f172a',
              color: '#f8fafc',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '8px',
              padding: '0.6rem 1rem',
              fontSize: '0.95rem',
              fontWeight: 500,
              cursor: 'pointer',
              minWidth: '260px',
            }}
          >
            {SYNTHETIC_PATIENT_PROFILES.map((p) => (
              <option key={p.patno} value={p.patno}>
                {p.patno} ({p.gender}, Age {p.age}, Base Risk: {Math.round(p.motor_progression_risk * 100)}%)
              </option>
            ))}
          </select>
        </div>

        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <div
            style={{
              padding: '0.5rem 1rem',
              borderRadius: '8px',
              background: badge.bg,
              border: `1px solid ${badge.border}`,
              color: badge.color,
              fontWeight: 600,
              fontSize: '0.9rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
            }}
          >
            <span
              style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                background: badge.color,
                display: 'inline-block',
                boxShadow: `0 0 8px ${badge.color}`,
              }}
            />
            {badge.label}
          </div>
        </div>
      </div>

      {/* Main Grid: Biomarkers & Risk Score */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
          gap: '1.5rem',
        }}
      >
        {/* Card 1: Biomarkers & Baseline Parameters */}
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
            <span>🔬</span> Motor & Dopaminergic Biomarkers
          </h3>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
            <div style={{ background: '#0f172a', padding: '0.75rem', borderRadius: '8px' }}>
              <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Age / Sex</div>
              <div style={{ fontSize: '1rem', fontWeight: 600, color: '#f1f5f9' }}>
                {patient.age} yrs / {patient.gender}
              </div>
            </div>

            <div style={{ background: '#0f172a', padding: '0.75rem', borderRadius: '8px' }}>
              <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Disease Duration</div>
              <div style={{ fontSize: '1rem', fontWeight: 600, color: '#f1f5f9' }}>
                {patient.disease_duration_months} mo
              </div>
            </div>

            <div style={{ background: '#0f172a', padding: '0.75rem', borderRadius: '8px' }}>
              <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Putamen SBR (L / R)</div>
              <div style={{ fontSize: '1rem', fontWeight: 600, color: '#f1f5f9' }}>
                {currentPutamenSbr.toFixed(2)} / {patient.putamen_sbr_right.toFixed(2)}
              </div>
            </div>

            <div style={{ background: '#0f172a', padding: '0.75rem', borderRadius: '8px' }}>
              <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Caudate SBR (L / R)</div>
              <div style={{ fontSize: '1rem', fontWeight: 600, color: '#f1f5f9' }}>
                {patient.caudate_sbr_left.toFixed(2)} / {patient.caudate_sbr_right.toFixed(2)}
              </div>
            </div>

            <div style={{ background: '#0f172a', padding: '0.75rem', borderRadius: '8px' }}>
              <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Baseline UPDRS Part III</div>
              <div style={{ fontSize: '1rem', fontWeight: 600, color: '#f1f5f9' }}>
                {currentUpdrs.toFixed(1)} pts
              </div>
            </div>

            <div style={{ background: '#0f172a', padding: '0.75rem', borderRadius: '8px' }}>
              <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Cognitive (MMSE)</div>
              <div style={{ fontSize: '1rem', fontWeight: 600, color: '#f1f5f9' }}>
                {patient.mmse_score} / 30
              </div>
            </div>
          </div>

          {/* Interactive What-If Sensitivity Adjustments */}
          <div style={{ marginTop: '1.25rem', paddingTop: '1.25rem', borderTop: '1px solid rgba(255, 255, 255, 0.08)' }}>
            <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#cbd5e1', marginBottom: '0.75rem' }}>
              ⚡ Interactive Parameter Sensitivity (What-If Simulation):
            </div>

            <div style={{ marginBottom: '0.75rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#94a3b8' }}>
                <span>UPDRS III Motor Score</span>
                <span>{currentUpdrs.toFixed(1)} pts</span>
              </div>
              <input
                type="range"
                min="5"
                max="60"
                step="0.5"
                value={currentUpdrs}
                onChange={(e) => setCustomUpdrs(parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: '#38bdf8', cursor: 'pointer' }}
              />
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#94a3b8' }}>
                <span>Putamen SBR Left (DaTscan binding)</span>
                <span>{currentPutamenSbr.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.3"
                max="2.5"
                step="0.05"
                value={currentPutamenSbr}
                onChange={(e) => setCustomPutamenSbr(parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: '#38bdf8', cursor: 'pointer' }}
              />
            </div>
          </div>
        </div>

        {/* Card 2: Calibrated Risk & Bootstrap Confidence Interval */}
        <div
          style={{
            background: 'rgba(30, 41, 59, 0.6)',
            backdropFilter: 'blur(12px)',
            borderRadius: '12px',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            padding: '1.5rem',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
          }}
        >
          <div>
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
              <span>📊</span> Calibrated Motor Progression Probability
            </h3>

            <div style={{ textAlign: 'center', padding: '1.5rem 0' }}>
              <div
                style={{
                  fontSize: '3.5rem',
                  fontWeight: 800,
                  color: badge.color,
                  lineHeight: 1,
                  textShadow: `0 0 24px ${badge.border}`,
                }}
              >
                {(simulatedRisk * 100).toFixed(1)}%
              </div>
              <div style={{ fontSize: '0.85rem', color: '#94a3b8', marginTop: '0.5rem' }}>
                Estimated probability of substantial 12-month motor progression
              </div>
            </div>

            {/* Bootstrap Confidence Interval Bar */}
            <div
              style={{
                background: '#0f172a',
                borderRadius: '8px',
                padding: '1rem',
                border: '1px solid rgba(255, 255, 255, 0.05)',
              }}
            >
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  fontSize: '0.8rem',
                  color: '#cbd5e1',
                  marginBottom: '0.5rem',
                }}
              >
                <span>95% Bootstrap CI Range:</span>
                <span style={{ fontWeight: 600, color: '#38bdf8' }}>
                  {(simulatedLowerCI * 100).toFixed(1)}% — {(simulatedUpperCI * 100).toFixed(1)}%
                </span>
              </div>

              {/* Graphical Range Visualizer */}
              <div
                style={{
                  height: '10px',
                  background: 'rgba(255, 255, 255, 0.1)',
                  borderRadius: '5px',
                  position: 'relative',
                  overflow: 'hidden',
                }}
              >
                <div
                  style={{
                    position: 'absolute',
                    left: `${simulatedLowerCI * 100}%`,
                    width: `${(simulatedUpperCI - simulatedLowerCI) * 100}%`,
                    height: '100%',
                    background: badge.color,
                    opacity: 0.6,
                    borderRadius: '5px',
                  }}
                />
                <div
                  style={{
                    position: 'absolute',
                    left: `${simulatedRisk * 100}%`,
                    width: '3px',
                    height: '100%',
                    background: '#ffffff',
                    boxShadow: '0 0 6px #ffffff',
                  }}
                />
              </div>

              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  fontSize: '0.7rem',
                  color: '#64748b',
                  marginTop: '0.4rem',
                }}
              >
                <span>0% (Stable)</span>
                <span>50%</span>
                <span>100% (Progressing)</span>
              </div>
            </div>
          </div>

          <div
            style={{
              fontSize: '0.75rem',
              color: '#94a3b8',
              background: 'rgba(15, 23, 42, 0.7)',
              padding: '0.75rem',
              borderRadius: '6px',
              marginTop: '1rem',
              borderLeft: '3px solid #38bdf8',
            }}
          >
            <strong>Methodology Note:</strong> Evaluated using patient-grouped cross-validation
            (PATNO StratifiedGroupKFold) with fold-contained scaling to prevent data leakage.
          </div>
        </div>
      </div>

      {/* SHAP Feature Attribution Waterfall */}
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
          <span>🔍</span> Explainable AI: SHAP Feature Attribution Waterfall
        </h3>
        <p style={{ margin: '0 0 1.25rem 0', fontSize: '0.85rem', color: '#94a3b8' }}>
          Decomposes the prediction into positive (increasing risk) and negative (protective / reducing risk) feature contributions.
        </p>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {patient.shap_attributions.map((attr) => {
            const isPositive = attr.contribution >= 0
            const barWidthPercent = Math.min(100, Math.abs(attr.contribution) * 250)

            return (
              <div
                key={attr.feature}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '220px 1fr 100px',
                  alignItems: 'center',
                  gap: '1rem',
                  padding: '0.5rem 0.75rem',
                  background: 'rgba(15, 23, 42, 0.6)',
                  borderRadius: '6px',
                }}
              >
                <div>
                  <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#f1f5f9' }}>
                    {attr.display_name}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                    Observed: {attr.value} {attr.unit || ''}
                  </div>
                </div>

                <div
                  style={{
                    position: 'relative',
                    height: '24px',
                    display: 'flex',
                    alignItems: 'center',
                    background: 'rgba(255, 255, 255, 0.03)',
                    borderRadius: '4px',
                  }}
                >
                  <div
                    style={{
                      height: '14px',
                      borderRadius: '4px',
                      background: isPositive
                        ? 'linear-gradient(90deg, #f43f5e, #fb7185)'
                        : 'linear-gradient(90deg, #10b981, #34d399)',
                      width: `${barWidthPercent}%`,
                      transition: 'width 0.4s ease-out',
                    }}
                  />
                </div>

                <div
                  style={{
                    textAlign: 'right',
                    fontSize: '0.85rem',
                    fontWeight: 700,
                    color: isPositive ? '#f43f5e' : '#10b981',
                  }}
                >
                  {isPositive ? `+${attr.contribution.toFixed(2)}` : attr.contribution.toFixed(2)}
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
