/**
 * RiskBadge — colour-coded pill for case risk levels.
 */

import React from 'react'
import { StyleSheet, Text, View } from 'react-native'
import type { RiskLevel } from '../types'

const RISK_COLORS: Record<RiskLevel, { bg: string; text: string }> = {
  low: { bg: '#E8F5E9', text: '#2E7D32' },
  moderate: { bg: '#FFF8E1', text: '#F57F17' },
  high: { bg: '#FFF3E0', text: '#E65100' },
  critical: { bg: '#FFEBEE', text: '#B71C1C' },
}

interface RiskBadgeProps {
  level: RiskLevel
  size?: 'sm' | 'md'
}

export default function RiskBadge({ level, size = 'md' }: RiskBadgeProps) {
  const { bg, text } = RISK_COLORS[level]
  const fontSize = size === 'sm' ? 10 : 12

  return (
    <View style={[styles.badge, { backgroundColor: bg }]}>
      <Text style={[styles.label, { color: text, fontSize }]}>
        {level.toUpperCase()}
      </Text>
    </View>
  )
}

const styles = StyleSheet.create({
  badge: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 12,
    alignSelf: 'flex-start',
  },
  label: {
    fontWeight: '700',
    letterSpacing: 0.5,
  },
})
