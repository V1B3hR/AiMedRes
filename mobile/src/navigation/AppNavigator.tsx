/**
 * App Navigator — bottom tab navigation with a nested stack for Cases.
 */

import React from 'react'
import { Text } from 'react-native'
import { NavigationContainer } from '@react-navigation/native'
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs'
import { createNativeStackNavigator } from '@react-navigation/native-stack'
import type { RootTabParamList, CasesStackParamList } from '../types'
import DashboardScreen from '../screens/DashboardScreen'
import CasesScreen from '../screens/CasesScreen'
import CaseDetailScreen from '../screens/CaseDetailScreen'
import AlertsScreen from '../screens/AlertsScreen'

const Tab = createBottomTabNavigator<RootTabParamList>()
const CasesStack = createNativeStackNavigator<CasesStackParamList>()

function CasesNavigator() {
  return (
    <CasesStack.Navigator>
      <CasesStack.Screen
        name="CasesList"
        component={CasesScreen}
        options={{ title: 'Cases' }}
      />
      <CasesStack.Screen
        name="CaseDetail"
        component={CaseDetailScreen}
        options={{ title: 'Case Detail' }}
      />
    </CasesStack.Navigator>
  )
}

function TabIcon({ name, focused }: { name: string; focused: boolean }) {
  const icons: Record<string, string> = {
    Dashboard: '📊',
    Cases: '🗂️',
    Alerts: '🔔',
  }
  return (
    <Text style={{ fontSize: 20, opacity: focused ? 1 : 0.6 }}>
      {icons[name] ?? '●'}
    </Text>
  )
}

export default function AppNavigator() {
  return (
    <NavigationContainer>
      <Tab.Navigator
        screenOptions={({ route }) => ({
          tabBarIcon: ({ focused }) => (
            <TabIcon name={route.name} focused={focused} />
          ),
          tabBarActiveTintColor: '#1565C0',
          tabBarInactiveTintColor: '#757575',
          headerStyle: { backgroundColor: '#1565C0' },
          headerTintColor: '#FFFFFF',
          headerTitleStyle: { fontWeight: '700' },
        })}
      >
        <Tab.Screen
          name="Dashboard"
          component={DashboardScreen}
          options={{ title: 'Dashboard' }}
        />
        <Tab.Screen
          name="Cases"
          component={CasesNavigator}
          options={{ headerShown: false, title: 'Cases' }}
        />
        <Tab.Screen
          name="Alerts"
          component={AlertsScreen}
          options={{ title: 'Alerts' }}
        />
      </Tab.Navigator>
    </NavigationContainer>
  )
}
