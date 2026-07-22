/**
 * MetricCard — single KPI tile for the Dashboard screen.
 */

import React from 'react'
import { StyleSheet, Text, View } from 'react-native'

interface MetricCardProps {
  label: string
  value: string | number
  unit?: string
  accent?: string
}

export default function MetricCard({
  label,
  value,
  unit,
  accent = '#1565C0',
}: MetricCardProps) {
  return (
    <View style={[styles.card, { borderLeftColor: accent }]}>
      <Text style={styles.label} numberOfLines={2}>
        {label}
      </Text>
      <View style={styles.valueRow}>
        <Text style={[styles.value, { color: accent }]}>{value}</Text>
        {unit ? <Text style={styles.unit}>{unit}</Text> : null}
      </View>
    </View>
  )
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: '#FFFFFF',
    borderRadius: 8,
    padding: 14,
    flex: 1,
    margin: 6,
    borderLeftWidth: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.06,
    shadowRadius: 2,
    elevation: 2,
  },
  label: {
    fontSize: 12,
    color: '#757575',
    marginBottom: 6,
    lineHeight: 16,
  },
  valueRow: {
    flexDirection: 'row',
    alignItems: 'flex-end',
  },
  value: {
    fontSize: 26,
    fontWeight: '700',
  },
  unit: {
    fontSize: 13,
    color: '#9E9E9E',
    marginLeft: 4,
    marginBottom: 3,
  },
})
