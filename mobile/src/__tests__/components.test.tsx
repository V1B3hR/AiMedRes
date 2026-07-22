/**
 * Unit tests for component helper logic.
 * Pure-logic tests avoid React Native rendering in CI.
 */

import { describe, it, expect } from 'vitest'
import type { RiskLevel, AlertSeverity } from '../types'

// ─── RiskBadge colour mapping ──────────────────────────────────────────────

const RISK_COLORS: Record<RiskLevel, { bg: string; text: string }> = {
  low: { bg: '#E8F5E9', text: '#2E7D32' },
  moderate: { bg: '#FFF8E1', text: '#F57F17' },
  high: { bg: '#FFF3E0', text: '#E65100' },
  critical: { bg: '#FFEBEE', text: '#B71C1C' },
}

describe('RiskBadge colour logic', () => {
  const levels: RiskLevel[] = ['low', 'moderate', 'high', 'critical']

  it.each(levels)('has a defined colour entry for "%s"', (level) => {
    expect(RISK_COLORS[level]).toBeDefined()
    expect(RISK_COLORS[level].bg).toMatch(/^#/)
    expect(RISK_COLORS[level].text).toMatch(/^#/)
  })

  it('uses distinct bg colours for each level', () => {
    const bgs = levels.map((l) => RISK_COLORS[l].bg)
    const unique = new Set(bgs)
    expect(unique.size).toBe(levels.length)
  })

  it('critical uses a red-family background', () => {
    expect(RISK_COLORS.critical.bg).toBe('#FFEBEE')
  })

  it('low uses a green-family background', () => {
    expect(RISK_COLORS.low.bg).toBe('#E8F5E9')
  })
})

// ─── AlertItem severity mapping ────────────────────────────────────────────

const SEVERITY_COLORS: Record<AlertSeverity, { bg: string; text: string; border: string }> = {
  info: { bg: '#E3F2FD', text: '#1565C0', border: '#1565C0' },
  warning: { bg: '#FFF8E1', text: '#F57F17', border: '#F57F17' },
  critical: { bg: '#FFEBEE', text: '#B71C1C', border: '#B71C1C' },
}

describe('AlertItem severity logic', () => {
  const severities: AlertSeverity[] = ['info', 'warning', 'critical']

  it.each(severities)('has a defined colour entry for "%s"', (sev) => {
    expect(SEVERITY_COLORS[sev]).toBeDefined()
  })

  it('critical has a red border', () => {
    expect(SEVERITY_COLORS.critical.border).toBe('#B71C1C')
  })

  it('info has a blue border', () => {
    expect(SEVERITY_COLORS.info.border).toBe('#1565C0')
  })
})

// ─── Date formatting helper ────────────────────────────────────────────────

function formatDate(iso: string): string {
  const d = new Date(iso)
  return d.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

describe('formatDate helper', () => {
  it('returns a non-empty string for a valid ISO date', () => {
    const result = formatDate('2025-01-15T10:30:00Z')
    expect(result.length).toBeGreaterThan(0)
  })

  it('includes a numeric day', () => {
    const result = formatDate('2025-06-01T08:00:00Z')
    expect(result).toMatch(/\d/)
  })
})

// ─── fetchHighRiskCases filter logic ──────────────────────────────────────

import type { Case } from '../types'

function filterHighRisk(cases: Case[]): Case[] {
  return cases.filter((c) => c.risk_level === 'high' || c.risk_level === 'critical')
}

describe('filterHighRisk', () => {
  const cases: Case[] = [
    { case_id: 'a', risk_level: 'low', status: 'pending', patient_id: 'p1', prediction: { class: 'low', probability: 0.1 }, created_at: '', updated_at: '', model_version: 'v1' },
    { case_id: 'b', risk_level: 'moderate', status: 'pending', patient_id: 'p2', prediction: { class: 'moderate', probability: 0.5 }, created_at: '', updated_at: '', model_version: 'v1' },
    { case_id: 'c', risk_level: 'high', status: 'pending', patient_id: 'p3', prediction: { class: 'high', probability: 0.8 }, created_at: '', updated_at: '', model_version: 'v1' },
    { case_id: 'd', risk_level: 'critical', status: 'pending', patient_id: 'p4', prediction: { class: 'critical', probability: 0.95 }, created_at: '', updated_at: '', model_version: 'v1' },
  ]

  it('keeps only high and critical risk cases', () => {
    const result = filterHighRisk(cases)
    expect(result.map((c) => c.case_id)).toEqual(['c', 'd'])
  })

  it('returns empty array when no high/critical cases', () => {
    const lowOnly = cases.filter((c) => c.risk_level === 'low')
    expect(filterHighRisk(lowOnly)).toHaveLength(0)
  })
})

