/**
 * Unit tests for the API client module.
 */

import { describe, it, expect, beforeEach } from 'vitest'
import { setAuthToken, getAuthToken } from '../api/client'

describe('API client token management', () => {
  beforeEach(() => {
    setAuthToken(null)
  })

  it('starts with null token', () => {
    expect(getAuthToken()).toBeNull()
  })

  it('stores and retrieves a token', () => {
    setAuthToken('test-jwt-token')
    expect(getAuthToken()).toBe('test-jwt-token')
  })

  it('clears token when set to null', () => {
    setAuthToken('some-token')
    setAuthToken(null)
    expect(getAuthToken()).toBeNull()
  })

  it('overwrites existing token', () => {
    setAuthToken('old-token')
    setAuthToken('new-token')
    expect(getAuthToken()).toBe('new-token')
  })
})
