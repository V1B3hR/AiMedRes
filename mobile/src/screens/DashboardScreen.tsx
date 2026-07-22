/**
 * Dashboard Screen — KPI overview and quick stats.
 */

import React, { useCallback, useState } from 'react'
import {
  ActivityIndicator,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native'
import { fetchDashboardStats } from '../api/dashboard'
import MetricCard from '../components/MetricCard'
import type { DashboardStats } from '../types'
import { useFocusEffect } from '@react-navigation/native'

export default function DashboardScreen() {
  const [stats, setStats] = useState<DashboardStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async (isRefresh = false) => {
    if (isRefresh) setRefreshing(true)
    else setLoading(true)
    setError(null)
    try {
      const data = await fetchDashboardStats()
      setStats(data)
    } catch {
      setError('Unable to load dashboard stats. Check your connection.')
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }, [])

  useFocusEffect(
    useCallback(() => {
      load()
    }, [load])
  )

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#1565C0" />
      </View>
    )
  }

  if (error) {
    return (
      <View style={styles.center}>
        <Text style={styles.errorText}>{error}</Text>
      </View>
    )
  }

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={() => load(true)} />
      }
    >
      <Text style={styles.sectionTitle}>Case Queue</Text>
      <View style={styles.row}>
        <MetricCard
          label="Pending"
          value={stats?.pending_cases ?? 0}
          accent="#F57F17"
        />
        <MetricCard
          label="In Review"
          value={stats?.in_review_cases ?? 0}
          accent="#1565C0"
        />
      </View>
      <View style={styles.row}>
        <MetricCard
          label="Completed Today"
          value={stats?.completed_today ?? 0}
          accent="#2E7D32"
        />
        <MetricCard
          label="High Risk"
          value={stats?.high_risk_cases ?? 0}
          accent="#B71C1C"
        />
      </View>

      <Text style={styles.sectionTitle}>System Health</Text>
      <View style={styles.row}>
        <MetricCard
          label="Critical Alerts"
          value={stats?.critical_alerts ?? 0}
          accent="#B71C1C"
        />
        <MetricCard
          label="Model Accuracy"
          value={
            stats?.model_accuracy != null
              ? `${(stats.model_accuracy * 100).toFixed(1)}`
              : '—'
          }
          unit="%"
          accent="#1565C0"
        />
      </View>
      <View style={styles.row}>
        <MetricCard
          label="Avg Processing Time"
          value={stats?.avg_processing_time_ms ?? '—'}
          unit="ms"
          accent="#6A1B9A"
        />
        <View style={{ flex: 1, margin: 6 }} />
      </View>
    </ScrollView>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  content: {
    paddingVertical: 12,
    paddingHorizontal: 10,
  },
  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F5F5F5',
  },
  errorText: {
    color: '#C62828',
    fontSize: 14,
    textAlign: 'center',
    paddingHorizontal: 24,
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#616161',
    marginTop: 12,
    marginBottom: 4,
    marginLeft: 6,
    textTransform: 'uppercase',
    letterSpacing: 0.6,
  },
  row: {
    flexDirection: 'row',
  },
})
