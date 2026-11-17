// Soak test for the compare_online API - free (no-auth) endpoints only.
// Purpose: catch slow degradation (memory leaks, connection-pool exhaustion,
// cache churn) that a 3-minute load spike cannot show.
//
// Run: k6 run tests/load/soak.js                      # default: 5m warmup + 50m steady + 5m cooldown = 60m
//      BASE_URL=http://localhost:8000 k6 run tests/load/soak.js
//      SOAK_VUS=2 SOAK_STEADY_MIN=10 k6 run tests/load/soak.js   # smoke variant (~20m)
//
// Gates (same SLO as tests/load/k6_script.js and env SLO_* in
// .github/workflows/load-test.yml, see docs/LOAD-BASELINE.md):
//   - http_req_failed rate < 0.01 (1%)
//   - http_req_duration p(95) < 800 ms (whole run, every stage)
//   - checks pass rate > 0.99 (99%)
//   - cooldown-stage latency p(95) < 400 ms: tightened end-of-run budget.
//     k6 cannot read the server's RSS, so a leak/degradation is detected as
//     the last 5 minutes being slower than the tightened late-run budget.
// k6 exits with code 99 when any threshold fails - that fails the CI job.

import http from 'k6/http';
import { check } from 'k6';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const SOAK_VUS = Number(__ENV.SOAK_VUS || 5);
const WARMUP_MIN = Number(__ENV.SOAK_WARMUP_MIN || 5);
const STEADY_MIN = Number(__ENV.SOAK_STEADY_MIN || 50);
const COOLDOWN_MIN = Number(__ENV.SOAK_COOLDOWN_MIN || 5);

export const options = {
  scenarios: {
    soak_warmup: {
      executor: 'constant-vus',
      vus: SOAK_VUS,
      duration: `${WARMUP_MIN}m`,
      tags: { stage: 'warmup' },
    },
    soak_steady: {
      executor: 'constant-vus',
      vus: SOAK_VUS,
      duration: `${STEADY_MIN}m`,
      tags: { stage: 'steady' },
      startTime: `${WARMUP_MIN}m`,
    },
    soak_cooldown: {
      executor: 'constant-vus',
      vus: 1,
      duration: `${COOLDOWN_MIN}m`,
      tags: { stage: 'cooldown' },
      startTime: `${WARMUP_MIN + STEADY_MIN}m`,
    },
  },
  thresholds: {
    // Общепрогонные SLO - те же числа, что в k6_script.js и workflow env SLO_*.
    http_req_failed: ['rate<0.01'],
    http_req_duration: ['p(95)<800'],
    checks: ['rate>0.99'],
    // Стадийные срезы: рост задержки к концу прогона = деградация.
    'http_req_duration{stage:warmup}': ['p(95)<800'],
    'http_req_duration{stage:steady}': ['p(95)<800'],
    // Финальный контроль после 60 минут нагрузки держим вдвое жёстче
    // (p95 < 400 ms на 1 VU): если память/пулы потекли, последние минуты
    // не влезут в бюджет даже при одном виртуальном пользователе.
    'http_req_duration{stage:cooldown}': ['p(95)<400'],
  },
};

export default function () {
  const health = http.get(`${BASE_URL}/api/health`, {
    tags: { name: 'GET /api/health', stage_probe: 'soak' },
  });
  check(health, {
    'GET /api/health returns 200': (r) => r.status === 200,
    'health body says ok': (r) => {
      try {
        return JSON.parse(r.body).status === 'ok';
      } catch (e) {
        return false;
      }
    },
  });

  const root = http.get(`${BASE_URL}/`, {
    tags: { name: 'GET /', stage_probe: 'soak' },
  });
  check(root, {
    'GET / returns 200': (r) => r.status === 200,
  });
}

// Пишем machine-readable итог в файл артефактом (тот же приём, что
// --summary-export в основном сценарии), чтобы "Check thresholds" шаг
// мог прогнать jq-гейт и по soak-отчёту.
export function handleSummary(data) {
  return {
    'tests/load/reports/soak-summary.json': JSON.stringify(data),
    stdout: `soak done: ${SOAK_VUS} VU, ${WARMUP_MIN}+${STEADY_MIN}+${COOLDOWN_MIN} min total\n`,
  };
}
