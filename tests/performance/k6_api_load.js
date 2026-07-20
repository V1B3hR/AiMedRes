import http from 'k6/http'
import { check, sleep } from 'k6'

const apiBaseUrl = __ENV.API_BASE_URL || 'http://localhost:8080'
const healthPath = __ENV.HEALTH_PATH || '/health'
const vus = Number(__ENV.K6_VUS || 10)
const duration = __ENV.K6_DURATION || '30s'

export const options = {
  vus,
  duration,
  thresholds: {
    http_req_failed: ['rate<0.05'],
    http_req_duration: ['p(95)<1000'],
  },
}

export default function () {
  const response = http.get(`${apiBaseUrl}${healthPath}`, {
    headers: { Accept: 'application/json' },
    tags: { endpoint: 'health' },
  })

  check(response, {
    'health status is 200': (r) => r.status === 200,
    'health latency under 1000ms': (r) => r.timings.duration < 1000,
  })

  sleep(1)
}
