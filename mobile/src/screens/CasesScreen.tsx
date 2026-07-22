/**
 * Cases Screen — filterable list of clinical cases.
 * Navigates to CaseDetailScreen on tap.
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
import { useNavigation } from '@react-navigation/native'
import type { NativeStackNavigationProp } from '@react-navigation/native-stack'
import { fetchCases } from '../api/cases'
import CaseCard from '../components/CaseCard'
import type { Case, CaseStatus, CasesStackParamList } from '../types'
import { useFocusEffect } from '@react-navigation/native'

type Nav = NativeStackNavigationProp<CasesStackParamList, 'CasesList'>

const STATUS_FILTERS: Array<{ label: string; value?: CaseStatus }> = [
  { label: 'All' },
  { label: 'Pending', value: 'pending' },
  { label: 'In Review', value: 'in_review' },
  { label: 'Completed', value: 'completed' },
]

export default function CasesScreen() {
  const navigation = useNavigation<Nav>()
  const [cases, setCases] = useState<Case[]>([])
  const [filter, setFilter] = useState<CaseStatus | undefined>(undefined)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [total, setTotal] = useState(0)

  const load = useCallback(
    async (isRefresh = false) => {
      if (isRefresh) setRefreshing(true)
      else setLoading(true)
      setError(null)
      try {
        const data = await fetchCases(filter)
        setCases(data.cases)
        setTotal(data.total)
      } catch {
        setError('Unable to load cases.')
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

  return (
    <View style={styles.container}>
      {/* Filter bar */}
      <View style={styles.filterBar}>
        {STATUS_FILTERS.map((f) => (
          <TouchableOpacity
            key={f.label}
            style={[styles.chip, filter === f.value && styles.chipActive]}
            onPress={() => setFilter(f.value)}
            accessibilityRole="button"
            accessibilityState={{ selected: filter === f.value }}
          >
            <Text
              style={[styles.chipText, filter === f.value && styles.chipTextActive]}
            >
              {f.label}
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
          data={cases}
          keyExtractor={(item) => item.case_id}
          renderItem={({ item }) => (
            <CaseCard
              item={item}
              onPress={(c) =>
                navigation.navigate('CaseDetail', { caseId: c.case_id })
              }
            />
          )}
          refreshControl={
            <RefreshControl refreshing={refreshing} onRefresh={() => load(true)} />
          }
          ListEmptyComponent={
            <View style={styles.center}>
              <Text style={styles.emptyText}>No cases found.</Text>
            </View>
          }
          ListHeaderComponent={
            <Text style={styles.totalText}>{total} case{total !== 1 ? 's' : ''}</Text>
          }
          contentContainerStyle={cases.length === 0 ? styles.emptyContainer : undefined}
        />
      )}
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
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
  chipActive: {
    backgroundColor: '#1565C0',
    borderColor: '#1565C0',
  },
  chipText: {
    fontSize: 13,
    color: '#757575',
  },
  chipTextActive: {
    color: '#FFFFFF',
    fontWeight: '600',
  },
  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 40,
  },
  emptyContainer: {
    flexGrow: 1,
  },
  emptyText: {
    fontSize: 15,
    color: '#9E9E9E',
  },
  errorText: {
    color: '#C62828',
    fontSize: 14,
    textAlign: 'center',
    paddingHorizontal: 24,
  },
  totalText: {
    fontSize: 12,
    color: '#9E9E9E',
    marginTop: 8,
    marginLeft: 16,
    marginBottom: 4,
  },
})
