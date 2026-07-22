/**
 * CaseCard — compact summary card used in the cases list.
 */

import React from 'react'
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native'
import type { Case } from '../types'
import RiskBadge from './RiskBadge'

interface CaseCardProps {
  item: Case
  onPress: (item: Case) => void
}

function formatDate(iso: string): string {
  const d = new Date(iso)
  return d.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export default function CaseCard({ item, onPress }: CaseCardProps) {
  return (
    <TouchableOpacity
      style={styles.card}
      onPress={() => onPress(item)}
      accessibilityRole="button"
      accessibilityLabel={`Case ${item.case_id}, risk ${item.risk_level}`}
    >
      <View style={styles.header}>
        <Text style={styles.caseId} numberOfLines={1}>
          Case {item.case_id}
        </Text>
        <RiskBadge level={item.risk_level} size="sm" />
      </View>

      <View style={styles.row}>
        <Text style={styles.label}>Patient</Text>
        <Text style={styles.value}>{item.patient_id}</Text>
      </View>

      <View style={styles.row}>
        <Text style={styles.label}>Status</Text>
        <Text style={[styles.value, styles.status]}>
          {item.status.replace('_', ' ')}
        </Text>
      </View>

      <View style={styles.row}>
        <Text style={styles.label}>Model</Text>
        <Text style={styles.value}>{item.model_version}</Text>
      </View>

      <View style={styles.row}>
        <Text style={styles.label}>Probability</Text>
        <Text style={styles.value}>
          {(item.prediction.probability * 100).toFixed(1)}%
        </Text>
      </View>

      <Text style={styles.timestamp}>{formatDate(item.created_at)}</Text>
    </TouchableOpacity>
  )
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: '#FFFFFF',
    borderRadius: 8,
    padding: 14,
    marginVertical: 6,
    marginHorizontal: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.08,
    shadowRadius: 3,
    elevation: 2,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  caseId: {
    fontSize: 15,
    fontWeight: '600',
    color: '#1A237E',
    flex: 1,
    marginRight: 8,
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginVertical: 2,
  },
  label: {
    fontSize: 13,
    color: '#757575',
  },
  value: {
    fontSize: 13,
    color: '#212121',
    fontWeight: '500',
  },
  status: {
    textTransform: 'capitalize',
  },
  timestamp: {
    fontSize: 11,
    color: '#BDBDBD',
    marginTop: 6,
    textAlign: 'right',
  },
})
