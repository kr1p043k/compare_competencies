# n8n Integration Guide

## 1. Quick Start

### 1.1 Configure .env
```env
# Обязательно для n8n
N8N_API_KEY=your-secret-api-key-here
N8N_WEBHOOK_SECRET=your-webhook-secret-here

# Для Telegram уведомлений
TG_BOT_TOKEN=your-bot-token
TG_CHAT_ID=your-chat-id
```

### 1.2 Start API
```bash
uvicorn src.api_pkg:app --host 0.0.0.0 --port 8000
```

### 1.3 n8n Setup
- Create **Generic Credential** → HTTP Header Auth:
  - Header Name: `Authorization`
  - Header Value: `Bearer <N8N_API_KEY>`
- All HTTP Request nodes use this credential
- Base URL: `http://localhost:8000` (dev) or `https://your-domain.com` (prod)

---

## 2. API Endpoints — n8n Reference

Full list in `src/api_pkg/n8n.py`. Key groups:

### Monitoring
| Method | Path | Rate | Use |
|--------|------|------|-----|
| GET | `/health` | — | Check service is alive |
| GET | `/ready` | — | All components ready |
| GET | `/api/status` | — | Full system status |

### Profiles + Recommendations (core)
| Method | Path | Rate | Use |
|--------|------|------|-----|
| GET | `/api/profiles/compare` | 20/min | Evaluate all profiles |
| GET | `/api/recommendations/{profile}` | 30/min | LTR recommendations |
| GET | `/api/profiles/{profile}/profession-evaluation` | 30/min | KRM profession eval |

### Pipeline
| Method | Path | Rate | Use |
|--------|------|------|-----|
| POST | `/api/pipeline/{action}` | 5/min | Run: full-cycle, rebuild, train-model, gap-analysis |
| GET  | `/api/pipeline/status` | 30/min | Check artifacts |
| GET  | `/api/pipeline/tasks` | 30/min | Task history |
| WS   | `/api/pipeline/ws` | — | Real-time progress |

---

## 3. Webhooks (n8n → API)

### 3.1 Endpoints
| Path | Event | Body |
|------|-------|------|
| `POST /api/n8n/webhook/student-created` | New student from n8n | `{profile_name, skills, target_level}` |
| `POST /api/n8n/webhook/pipeline-completed` | External pipeline done | `{task_id, status, artifacts}` |
| `POST /api/n8n/webhook/alert` | Alert from n8n | `{type, severity, message, data?}` |
| `GET /api/n8n/webhooks` | List recent events (max 50) | — |

All webhooks require header: `X-N8N-Webhook-Secret: <N8N_WEBHOOK_SECRET>`

### 3.2 n8n → API Workflow
```
[Trigger] → [HTTP Request: POST /api/n8n/webhook/...] → [Done]
```

---

## 4. Workflow Templates

Pre-exported JSON templates in `src/n8n/workflows/`:

| File | Description | Trigger |
|------|-------------|---------|
| `nightly_pipeline.json` | Еженочный пайплайн + Telegram | Schedule (24h) |
| `trend_alert.json` | Алёрт при скачке тренда >25% | Schedule (weekly) |
| `student_onboarding.json` | Приём студента через webhook | Webhook |

Import in n8n: **Workflows → Add → Import from File**

---

## 5. Common n8n Workflow Patterns

### 5.1 Nightly Pipeline + Notify
```
Schedule (daily 02:00)
  → HTTP POST /api/pipeline/full-cycle?skip_collection=true&run_gap_analysis=true
  → Wait 5 min
  → HTTP GET /api/pipeline/active
  → IF active → Wait 5 min → loop
  → HTTP GET /api/pipeline/status
  → Telegram: "Pipeline completed"
```

### 5.2 Profile Change Monitor
```
Schedule (weekly)
  → HTTP GET /api/profiles/compare
  → Compare with previous run (n8n data store)
  → IF coverage dropped >5% → Telegram alert
```

### 5.3 Gap Analysis on Demand
```
Webhook (from admin panel)
  → HTTP POST /api/pipeline/gap-analysis?run_gap_analysis=true
  → HTTP GET /api/pipeline/gap-progress/{task_id} (loop)
  → HTTP GET /api/results/summary
  → Email report
```

---

## 6. n8n Environment Variables

Set in n8n **Settings → Environment Variables**:
```env
BASE_URL=http://localhost:8000
TG_CHAT_ID=123456789
```

Or use n8n credentials for the HTTP Header Auth.

---

## 7. Architecture

```
┌─────────────┐     HTTP (Bearer)     ┌────────────────────┐
│   n8n       │ ────────────────────> │ FastAPI (this app) │
│  (external) │ <──────────────────── │ :8000              │
└─────────────┘     JSON response     └────────┬───────────┘
                                               │
                                     ┌─────────▼─────────┐
                                     │  n8n Webhooks      │
                                     │  /api/n8n/webhook/*│
                                     └───────────────────┘
```

- n8n is **external** — runs separately (Docker, n8n.cloud, or local)
- Communication is **pull-based** (n8n polls API) + **webhook** (n8n pushes events)
- Auth via **Bearer token** (N8N_API_KEY)
- Rate limits per endpoint prevent abuse

---

## 8. Security

1. **N8N_API_KEY** — required for all API calls from n8n
2. **N8N_WEBHOOK_SECRET** — required for webhook endpoints
3. Both stored in `.env`, never committed
4. Rate limits: 2-60 req/min per endpoint
5. All n8n traffic should go through HTTPS in production
