/**
 * Alerts Screen — lists clinical alerts with acknowledge action.
 */

import React, { useCallback, useState } from 'react'
import {
  ActivityIndicator,
  FlatList,
  RefreshControl,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native'
import { useFocusEffect } from '@react-navigation/native'
import { acknowledgeAlert, fetchAlerts } from '../api/alerts'
import AlertItem from '../components/AlertItem'
import type { ClinicalAlert } from '../types'

type Filter = 'all' | 'unacknowledged'

export default function AlertsScreen() {
  const [alerts, setAlerts] = useState<ClinicalAlert[]>([])
  const [unacknowledgedCount, setUnacknowledgedCount] = useState(0)
  const [filter, setFilter] = useState<Filter>('unacknowledged')
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(
    async (isRefresh = false) => {
      if (isRefresh) setRefreshing(true)
      else setLoading(true)
      setError(null)
      try {
        const data = await fetchAlerts(
          filter === 'unacknowledged' ? false : undefined
        )
        setAlerts(data.alerts)
        setUnacknowledgedCount(data.unacknowledged_count)
      } catch {
        setError('Unable to load alerts.')
      } finally {
        setLoading(false)
        setRefreshing(false)
      }
    },
    [filter]
  )

  useFocusEffect(
    useCallback(() => {
      load()
    }, [load])
  )

  const handleAcknowledge = async (alertId: string) => {
    try {
      await acknowledgeAlert(alertId)
      setAlerts((prev) =>
        prev.map((a) =>
          a.alert_id === alertId
            ? { ...a, acknowledged: true, acknowledged_at: new Date().toISOString() }
            : a
        )
      )
      setUnacknowledgedCount((prev) => Math.max(0, prev - 1))
    } catch {
      // silently fail — user can retry via refresh
    }
  }

  return (
    <View style={styles.container}>
      {/* Unacknowledged count banner */}
      {unacknowledgedCount > 0 && (
        <View style={styles.banner}>
          <Text style={styles.bannerText}>
            {unacknowledgedCount} unacknowledged alert
            {unacknowledgedCount !== 1 ? 's' : ''}
          </Text>
        </View>
      )}

      {/* Filter bar */}
      <View style={styles.filterBar}>
        {(['unacknowledged', 'all'] as Filter[]).map((f) => (
          <TouchableOpacity
            key={f}
            style={[styles.chip, filter === f && styles.chipActive]}
            onPress={() => setFilter(f)}
            accessibilityRole="button"
            accessibilityState={{ selected: filter === f }}
          >
            <Text
              style={[styles.chipText, filter === f && styles.chipTextActive]}
            >
              {f === 'unacknowledged' ? 'Unacknowledged' : 'All'}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      {loading ? (
        <View style={styles.center}>
          <ActivityIndicator size="large" color="#1565C0" />
        </View>
      ) : error ? (
        <View style={styles.center}>
          <Text style={styles.errorText}>{error}</Text>
        </View>
      ) : (
        <FlatList
          data={alerts}
          keyExtractor={(item) => item.alert_id}
          renderItem={({ item }) => (
            <AlertItem alert={item} onAcknowledge={handleAcknowledge} />
          )}
          refreshControl={
            <RefreshControl refreshing={refreshing} onRefresh={() => load(true)} />
          }
          ListEmptyComponent={
            <View style={styles.center}>
              <Text style={styles.emptyText}>
                {filter === 'unacknowledged' ? 'All alerts acknowledged.' : 'No alerts.'}
              </Text>
            </View>
          }
          contentContainerStyle={alerts.length === 0 ? styles.emptyContainer : undefined}
        />
      )}
    </View>
  )
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F5F5F5' },
  banner: {
    backgroundColor: '#B71C1C',
    paddingVertical: 8,
    paddingHorizontal: 16,
    alignItems: 'center',
  },
  bannerText: { color: '#FFFFFF', fontWeight: '700', fontSize: 13 },
  filterBar: {
    flexDirection: 'row',
    paddingHorizontal: 12,
    paddingVertical: 10,
    backgroundColor: '#FFFFFF',
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
    gap: 8,
  },
  chip: {
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: 16,
    backgroundColor: '#F5F5F5',
    borderWidth: 1,
    borderColor: '#E0E0E0',
  },
  chipActive: { backgroundColor: '#1565C0', borderColor: '#1565C0' },
  chipText: { fontSize: 13, color: '#757575' },
  chipTextActive: { color: '#FFFFFF', fontWeight: '600' },
  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 40,
  },
  emptyContainer: { flexGrow: 1 },
  emptyText: { fontSize: 15, color: '#9E9E9E' },
  errorText: {
    color: '#C62828',
    fontSize: 14,
    textAlign: 'center',
    paddingHorizontal: 24,
  },
})
