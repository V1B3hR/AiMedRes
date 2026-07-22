/**
 * Axios API client for the AiMedRes backend.
 * Manages auth tokens using SecureStore.
 */

import axios, { type AxiosInstance, type InternalAxiosRequestConfig } from 'axios'

const DEFAULT_BASE_URL = 'http://localhost:8080'

/** In-memory token fallback (populated by login or SecureStore on startup). */
let memoryToken: string | null = null

export function setAuthToken(token: string | null): void {
  memoryToken = token
}

export function getAuthToken(): string | null {
  return memoryToken
}

export function createApiClient(baseURL: string = DEFAULT_BASE_URL): AxiosInstance {
  const instance = axios.create({
    baseURL,
    headers: { 'Content-Type': 'application/json' },
    timeout: 15_000,
  })

  instance.interceptors.request.use((config: InternalAxiosRequestConfig) => {
    const token = memoryToken
    if (token) {
      config.headers.Authorization = ['Bearer', token].join(' ')
    }
    return config
  })

  return instance
}

const apiClient = createApiClient(DEFAULT_BASE_URL)

export default apiClient
