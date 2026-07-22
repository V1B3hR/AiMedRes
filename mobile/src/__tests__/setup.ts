/**
 * Test setup: mock React Native core modules so API and logic tests
 * can run in a plain Node / Vitest environment without a native runtime.
 */

import { vi } from 'vitest'

// Minimal React Native mock — only the primitives used in unit tests
vi.mock('react-native', () => ({
  StyleSheet: {
    create: (styles: Record<string, unknown>) => styles,
    hairlineWidth: 0.5,
  },
  View: ({ children }: { children?: unknown }) => children,
  Text: ({ children }: { children?: unknown }) => children,
  TouchableOpacity: ({ children }: { children?: unknown }) => children,
  TextInput: () => null,
  FlatList: () => null,
  ScrollView: ({ children }: { children?: unknown }) => children,
  ActivityIndicator: () => null,
  RefreshControl: () => null,
  Alert: { alert: vi.fn() },
  Platform: { OS: 'ios', select: (obj: Record<string, unknown>) => obj.ios ?? obj.default },
}))

// react-native-safe-area-context
vi.mock('react-native-safe-area-context', () => ({
  SafeAreaProvider: ({ children }: { children?: unknown }) => children,
  useSafeAreaInsets: () => ({ top: 0, right: 0, bottom: 0, left: 0 }),
}))

// expo-status-bar
vi.mock('expo-status-bar', () => ({
  StatusBar: () => null,
}))

// @react-navigation/native
vi.mock('@react-navigation/native', () => ({
  NavigationContainer: ({ children }: { children?: unknown }) => children,
  useNavigation: () => ({ navigate: vi.fn(), goBack: vi.fn() }),
  useRoute: () => ({ params: {} }),
  useFocusEffect: (cb: () => (() => void) | void) => cb(),
}))

// @react-navigation/bottom-tabs
vi.mock('@react-navigation/bottom-tabs', () => ({
  createBottomTabNavigator: () => ({
    Navigator: ({ children }: { children?: unknown }) => children,
    Screen: () => null,
  }),
}))

// @react-navigation/native-stack
vi.mock('@react-navigation/native-stack', () => ({
  createNativeStackNavigator: () => ({
    Navigator: ({ children }: { children?: unknown }) => children,
    Screen: () => null,
  }),
}))

