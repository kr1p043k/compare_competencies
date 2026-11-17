// Load test for the compare_online API - free (no-auth) endpoints only.
// Run: k6 run tests/load/k6_script.js
//      BASE_URL=https://staging.example.com k6 run tests/load/k6_script.js
//
// Primary CI gate: k6 exits with code 99 when a threshold below is breached.
// The numbers are the agreed SLO; they must match env SLO_* in
// .github/workflows/load-test.yml (secondary gate) - rationale:
// docs/LOAD-BASELINE.md ("SLO-пороги CI").

import http from 'k6/http';
import { check } from 'k6';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

export const options = {
  scenarios: {
    ramping: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '30s', target: 10 }, // ramp up 0 -> 10
        { duration: '2m', target: 10 },  // steady load
        { duration: '30s', target: 0 },  // ramp down to 0
      ],
      gracefulRampDown: '10s',
    },
  },
  thresholds: {
    // SLO (совпадает с env SLO_* в .github/workflows/load-test.yml):
    // p95 < 800 ms, доля ошибок < 1%, passed checks > 99%.
    http_req_failed: ['rate<0.01'],
    http_req_duration: ['p(95)<800'],
    checks: ['rate>0.99'],
  },
};

export default function () {
  const health = http.get(`${BASE_URL}/api/health`, {
    tags: { name: 'GET /api/health', endpoint: 'api_health' },
  });
  check(health, {
    'GET /api/health returns 200': (r) => r.status === 200,
  });

  const root = http.get(`${BASE_URL}/`, {
    tags: { name: 'GET /', endpoint: 'root' },
  });
  check(root, {
    'GET / returns 200': (r) => r.status === 200,
  });
}
