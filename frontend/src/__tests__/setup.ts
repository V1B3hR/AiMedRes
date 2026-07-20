/**
 * Vitest global test setup.
 *
 * Provides common mocks and utilities used across all unit tests.
 */

// Minimal localStorage mock (jsdom provides one but let's be explicit for clarity)
Object.defineProperty(window, 'localStorage', {
  value: (() => {
    let store: Record<string, string> = {}
    return {
      getItem: (key: string) => store[key] ?? null,
      setItem: (key: string, value: string) => { store[key] = value },
      removeItem: (key: string) => { delete store[key] },
      clear: () => { store = {} },
      get length() { return Object.keys(store).length },
      key: (idx: number) => Object.keys(store)[idx] ?? null,
    }
  })(),
  writable: true,
})

// Suppress noisy console.error in tests (re-enable per-test if needed)
vi.spyOn(console, 'error').mockImplementation(() => {})
