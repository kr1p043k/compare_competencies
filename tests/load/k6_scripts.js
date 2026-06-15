import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

const errorRate = new Rate('errors');
const gapDuration = new Trend('gap_duration');

export const options = {
  stages: [
    { duration: '30s', target: 20 },   // Ramp up
    { duration: '2m', target: 50 },    // Steady
    { duration: '30s', target: 100 },  // Spike
    { duration: '2m', target: 100 },   // High load
    { duration: '30s', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'],  // 95% запросов < 500ms
    errors: ['rate<0.05'],             // Ошибки < 5%
    gap_duration: ['p(95)<5000'],      // Gap-анализ < 5s
  },
};

const BASE_URL = 'http://localhost:8000';
const PROFILES = ['base', 'dc', 'top_dc'];
const QUERIES = ['python', 'java', 'data scientist', 'devops', 'sql'];

export default function () {
  // 1. Vacancies
  const vacRes = http.get(`${BASE_URL}/api/vacancies?query=${QUERIES[Math.floor(Math.random() * QUERIES.length)]}&limit=20`);
  check(vacRes, { 'vacancies status 200': (r) => r.status === 200 });
  errorRate.add(vacRes.status !== 200);
  
  // 2. Gap analysis (тяжёлый)
  const start = Date.now();
  const gapRes = http.post(`${BASE_URL}/api/gap-analysis`, JSON.stringify({
    student_profile: PROFILES[Math.floor(Math.random() * PROFILES.length)],
    region_id: 1,
    top_n: 10
  }), { headers: { 'Content-Type': 'application/json' } });
  gapDuration.add(Date.now() - start);
  check(gapRes, { 'gap status 200': (r) => r.status === 200 });
  
  // 3. Trends
  const trendsRes = http.get(`${BASE_URL}/api/trends?days=30`);
  check(trendsRes, { 'trends status 200': (r) => r.status === 200 });
  
  sleep(1);
}