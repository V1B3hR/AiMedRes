/**
 * Case Detail Screen — full case info with explainability and approve/reject actions.
 */

import React, { useCallback, useState } from 'react'
import {
  ActivityIndicator,
  Alert,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native'
import { useRoute, type RouteProp } from '@react-navigation/native'
import { approveCase, fetchCaseDetail } from '../api/cases'
import RiskBadge from '../components/RiskBadge'
import type { CaseDetail, CasesStackParamList } from '../types'
import { useFocusEffect } from '@react-navigation/native'

type Route = RouteProp<CasesStackParamList, 'CaseDetail'>

export default function CaseDetailScreen() {
  const { params } = useRoute<Route>()
  const [caseDetail, setCaseDetail] = useState<CaseDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [rationale, setRationale] = useState('')
  const [submitting, setSubmitting] = useState(false)

  useFocusEffect(
    useCallback(() => {
      let active = true
      setLoading(true)
      setError(null)
      fetchCaseDetail(params.caseId)
        .then((data) => { if (active) setCaseDetail(data) })
        .catch(() => { if (active) setError('Unable to load case details.') })
        .finally(() => { if (active) setLoading(false) })
      return () => { active = false }
    }, [params.caseId])
  )

  const handleAction = async (action: 'approve' | 'reject') => {
    if (!rationale.trim()) {
      Alert.alert('Rationale required', 'Please enter a rationale before submitting.')
      return
    }
    setSubmitting(true)
    try {
      await approveCase(params.caseId, action, rationale.trim())
      Alert.alert(
        'Success',
        `Case ${action === 'approve' ? 'approved' : 'rejected'} successfully.`
      )
      setRationale('')
    } catch {
      Alert.alert('Error', 'Failed to submit decision. Please try again.')
    } finally {
      setSubmitting(false)
    }
  }

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#1565C0" />
      </View>
    )
  }

  if (error || !caseDetail) {
    return (
      <View style={styles.center}>
        <Text style={styles.errorText}>{error ?? 'Case not found.'}</Text>
      </View>
    )
  }

  const { explainability } = caseDetail

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Header */}
      <View style={styles.headerCard}>
        <View style={styles.headerRow}>
          <Text style={styles.caseId}>Case {caseDetail.case_id}</Text>
          <RiskBadge level={caseDetail.risk_level} />
        </View>
        <Text style={styles.patientId}>Patient: {caseDetail.patient_id}</Text>
        {caseDetail.condition_type && (
          <Text style={styles.meta}>Condition: {caseDetail.condition_type}</Text>
        )}
      </View>

      {/* Prediction */}
      <Text style={styles.sectionTitle}>Prediction</Text>
      <View style={styles.card}>
        <Row label="Class" value={caseDetail.prediction.class} />
        <Row
          label="Probability"
          value={`${(caseDetail.prediction.probability * 100).toFixed(1)}%`}
        />
        {caseDetail.prediction.risk_score != null && (
          <Row
            label="Risk Score"
            value={caseDetail.prediction.risk_score.toFixed(3)}
          />
        )}
        <Row label="Model" value={caseDetail.model_version} />
        <Row label="Status" value={caseDetail.status.replace('_', ' ')} />
      </View>

      {/* Explainability */}
      {explainability && (
        <>
          <Text style={styles.sectionTitle}>Explainability</Text>
          <View style={styles.card}>
            <Row
              label="Confidence"
              value={`${(explainability.uncertainty.confidence * 100).toFixed(1)}%`}
            />
            <Row
              label="Total Uncertainty"
              value={explainability.uncertainty.total_uncertainty.toFixed(3)}
            />
          </View>

          {explainability.attributions.length > 0 && (
            <>
              <Text style={styles.sectionTitle}>Feature Attributions</Text>
              <View style={styles.card}>
                {explainability.attributions
                  .slice()
                  .sort((a, b) => b.importance - a.importance)
                  .map((attr, idx) => (
                    <View key={idx} style={styles.attrRow}>
                      <Text style={styles.attrFeature}>{attr.feature}</Text>
                      <View style={styles.attrBar}>
                        <View
                          style={[
                            styles.attrFill,
                            {
                              width: `${Math.min(attr.importance * 100, 100)}%`,
                            },
                          ]}
                        />
                      </View>
                      <Text style={styles.attrValue}>
                        {(attr.importance * 100).toFixed(0)}%
                      </Text>
                    </View>
                  ))}
              </View>
            </>
          )}
        </>
      )}

      {/* Decision */}
      {caseDetail.status === 'pending' || caseDetail.status === 'in_review' ? (
        <>
          <Text style={styles.sectionTitle}>Clinical Decision</Text>
          <View style={styles.card}>
            <Text style={styles.inputLabel}>Rationale *</Text>
            <TextInput
              style={styles.input}
              multiline
              numberOfLines={3}
              placeholder="Enter clinical rationale…"
              value={rationale}
              onChangeText={setRationale}
              accessibilityLabel="Clinical rationale input"
            />
            <View style={styles.buttonRow}>
              <TouchableOpacity
                style={[styles.button, styles.rejectButton]}
                onPress={() => handleAction('reject')}
                disabled={submitting}
                accessibilityRole="button"
                accessibilityLabel="Reject case"
              >
                <Text style={styles.buttonText}>
                  {submitting ? '…' : 'Reject'}
                </Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.button, styles.approveButton]}
                onPress={() => handleAction('approve')}
                disabled={submitting}
                accessibilityRole="button"
                accessibilityLabel="Approve case"
              >
                <Text style={styles.buttonText}>
                  {submitting ? '…' : 'Approve'}
                </Text>
              </TouchableOpacity>
            </View>
          </View>
        </>
      ) : null}
    </ScrollView>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.row}>
      <Text style={styles.rowLabel}>{label}</Text>
      <Text style={styles.rowValue}>{value}</Text>
    </View>
  )
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F5F5F5' },
  content: { paddingBottom: 32 },
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
  headerCard: {
    backgroundColor: '#1565C0',
    padding: 16,
    marginBottom: 4,
  },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  caseId: { fontSize: 18, fontWeight: '700', color: '#FFFFFF' },
  patientId: { fontSize: 14, color: '#BBDEFB' },
  meta: { fontSize: 13, color: '#90CAF9', marginTop: 2 },
  sectionTitle: {
    fontSize: 12,
    fontWeight: '600',
    color: '#757575',
    textTransform: 'uppercase',
    letterSpacing: 0.6,
    marginTop: 16,
    marginBottom: 4,
    marginHorizontal: 16,
  },
  card: {
    backgroundColor: '#FFFFFF',
    marginHorizontal: 16,
    borderRadius: 8,
    padding: 14,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.06,
    shadowRadius: 2,
    elevation: 1,
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 4,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#EEEEEE',
  },
  rowLabel: { fontSize: 13, color: '#757575' },
  rowValue: { fontSize: 13, color: '#212121', fontWeight: '500', textTransform: 'capitalize' },
  attrRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 5,
  },
  attrFeature: { fontSize: 12, color: '#424242', width: 120 },
  attrBar: {
    flex: 1,
    height: 6,
    backgroundColor: '#E0E0E0',
    borderRadius: 3,
    overflow: 'hidden',
    marginHorizontal: 8,
  },
  attrFill: { height: 6, backgroundColor: '#1565C0', borderRadius: 3 },
  attrValue: { fontSize: 12, color: '#757575', width: 32, textAlign: 'right' },
  inputLabel: { fontSize: 13, color: '#757575', marginBottom: 6 },
  input: {
    borderWidth: 1,
    borderColor: '#E0E0E0',
    borderRadius: 6,
    padding: 10,
    fontSize: 14,
    color: '#212121',
    minHeight: 72,
    textAlignVertical: 'top',
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    marginTop: 12,
    gap: 10,
  },
  button: {
    paddingHorizontal: 24,
    paddingVertical: 10,
    borderRadius: 6,
    minWidth: 90,
    alignItems: 'center',
  },
  approveButton: { backgroundColor: '#2E7D32' },
  rejectButton: { backgroundColor: '#C62828' },
  buttonText: { color: '#FFFFFF', fontWeight: '700', fontSize: 14 },
})
