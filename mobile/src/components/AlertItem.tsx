/**
 * AlertItem — single row for the Alerts list.
 */

import React from 'react'
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native'
import type { AlertSeverity, ClinicalAlert } from '../types'

const SEVERITY_COLORS: Record<AlertSeverity, { bg: string; text: string; border: string }> = {
  info: { bg: '#E3F2FD', text: '#1565C0', border: '#1565C0' },
  warning: { bg: '#FFF8E1', text: '#F57F17', border: '#F57F17' },
  critical: { bg: '#FFEBEE', text: '#B71C1C', border: '#B71C1C' },
}

interface AlertItemProps {
  alert: ClinicalAlert
  onAcknowledge: (alertId: string) => void
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

export default function AlertItem({ alert, onAcknowledge }: AlertItemProps) {
  const { bg, text, border } = SEVERITY_COLORS[alert.severity]

  return (
    <View
      style={[
        styles.container,
        { backgroundColor: bg, borderLeftColor: border },
        alert.acknowledged && styles.acknowledged,
      ]}
      accessibilityRole="alert"
    >
      <View style={styles.header}>
        <View style={styles.titleRow}>
          <Text style={[styles.severityLabel, { color: text }]}>
            {alert.severity.toUpperCase()}
          </Text>
          <Text style={styles.category}>
            {alert.category.replace(/_/g, ' ')}
          </Text>
        </View>
        {!alert.acknowledged && (
          <TouchableOpacity
            style={[styles.ackButton, { borderColor: border }]}
            onPress={() => onAcknowledge(alert.alert_id)}
            accessibilityRole="button"
            accessibilityLabel="Acknowledge alert"
          >
            <Text style={[styles.ackText, { color: text }]}>ACK</Text>
          </TouchableOpacity>
        )}
      </View>

      <Text style={styles.title}>{alert.title}</Text>
      <Text style={styles.message}>{alert.message}</Text>

      {alert.case_id && (
        <Text style={styles.meta}>Case: {alert.case_id}</Text>
      )}
      <Text style={styles.timestamp}>{formatDate(alert.created_at)}</Text>
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    borderLeftWidth: 4,
    borderRadius: 8,
    padding: 12,
    marginVertical: 5,
    marginHorizontal: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.06,
    shadowRadius: 2,
    elevation: 1,
  },
  acknowledged: {
    opacity: 0.55,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  titleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  severityLabel: {
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 0.5,
  },
  category: {
    fontSize: 11,
    color: '#757575',
    textTransform: 'capitalize',
  },
  ackButton: {
    borderWidth: 1,
    borderRadius: 4,
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  ackText: {
    fontSize: 11,
    fontWeight: '600',
  },
  title: {
    fontSize: 14,
    fontWeight: '600',
    color: '#212121',
    marginBottom: 4,
  },
  message: {
    fontSize: 13,
    color: '#424242',
    lineHeight: 18,
  },
  meta: {
    fontSize: 12,
    color: '#757575',
    marginTop: 6,
  },
  timestamp: {
    fontSize: 11,
    color: '#BDBDBD',
    marginTop: 4,
    textAlign: 'right',
  },
})
