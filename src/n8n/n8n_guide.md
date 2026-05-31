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
- Set **Environment Variable** in n8n Settings → Environment Variables:
  ```env
  BASE_URL=https://penny-coffee-considered-ala.trycloudflare.com
  ```
- Workflow templates use `{{ $env.BASE_URL }}` — обнови значение при смене туннеля

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

### Pipeline
| Method | Path | Rate | Use |
|--------|------|------|-----|
| POST | `/api/pipeline/{action}` | 5/min | Run: full-cycle, rebuild, train-model, gap-analysis |
| GET  | `/api/pipeline/status` | 30/min | Check artifacts |
| GET  | `/api/pipeline/tasks` | 30/min | Task history |
| WS   | `/api/pipeline/ws` | — | Real-time progress |

### Market & Trends
| Method | Path | Rate | Use |
|--------|------|------|-----|
| GET | `/api/market/top-skills` | 60/min | Top market skills |
| GET | `/api/market/skill/{skill}` | 60/min | Single skill detail |
| GET | `/api/trends` | 60/min | Trending skills analysis |

### Admin
| Method | Path | Rate | Use |
|--------|------|------|-----|
| GET | `/api/admin/export/excel` | 3/min | Export full report to Excel |
| GET | `/api/admin/export/full-report` | 3/min | Full analytics report XLSX |

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

| File | Description | Trigger | LLM | Postgres | Error Handler |
|------|-------------|---------|-----|----------|---------------|
| `nightly_pipeline.json` | Еженочный пайплайн + Telegram | Schedule (24h) | — | — | — |
| `trend_alert.json` | Алёрт при скачке тренда >25% | Schedule (weekly) | — | — | — |
| `student_onboarding.json` | Приём студента через webhook | Webhook | — | — | — |
| `weekly_report.json` | Еженедельный отчёт: 5 API → LLM → TG + Email + PG | Schedule (Mon 09:00) | Gemini/OpenAI | ✅ | ✅ |

### 4.1 Weekly Report Architecture

```
Schedule (Mon 09:00)
  → Health Check → IF Service Healthy
    → 5 параллельных запросов (Pipeline Status, Top Skills,
       Trends, Profiles Compare, Vacancy Stats)
    → Merge (combineByPosition)
    → Aggregate Report Data
    → Build Report Prompt
    → Switch LLM (Gemini / OpenAI по $env.LLM_PROVIDER)
    → Parse Report (JSON extraction + fallback)
    → Format Telegram (HTML)
    → Send Telegram ($env.TG_CHAT_ID)
    → Save Report to Postgres (таблица weekly_reports)
    → Build Email HTML + Fetch Excel Report
    → Send Email (HTML + XLSX вложение)
```

**Key improvements over legacy:**
- `$env.BASE_URL` вместо хардкода домена
- LLM-генерация отчёта (Gemini / OpenAI) вместо JavaScript форматировки
- Постгрес-персистентность (`weekly_reports` таблица)
- Error Trigger в `settings.errorWorkflow`
- Email с профессиональным HTML и Excel-вложением

Import in n8n: **Workflows → Add → Import from File**

Required credentials before import:
- **Generic Credential**: HTTP Header Auth (`Authorization: Bearer <N8N_API_KEY>`)
- **Google Palm API** (для Gemini Report)
- **OpenAI API** (опционально, для OpenAI Report)
- **SMTP** (для Send Email) — назвать `SFEDU SMTP`
- **Telegram Bot** (для Send Telegram)

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

### 5.4 Weekly Report (LLM-powered)
```
Schedule (Mon 09:00)
  → Health Check
  → IF Service Healthy
    → (true) 5 parallel API calls:
      → Pipeline Status, Top Skills, Trends,
        Profiles Compare, Vacancy Stats
    → (false) stop silently
  → Merge Data Sources (combineByPosition, 5 inputs)
  → Aggregate Report Data (Code: merge JSON)
  → Build Report Prompt (Code: assemble LLM prompt)
  → Switch LLM ($env.LLM_PROVIDER):
    → case "gemini": Gemini Report (Google Palm)
    → case "openai": OpenAI Report
    → default: Gemini Report
  → Parse Report (Code: extract JSON, fallback on fail)
  → Format Telegram (Code: HTML for Telegram)
  → Send Telegram to $env.TG_CHAT_ID (HTML parse_mode)
  → Save Report to Postgres (таблица weekly_reports)
  → Build Email HTML (Code: HTML template)
  → Fetch Excel Report (HTTP GET /api/admin/export/full-report)
  → Send Email (SFEDU SMTP, HTML + XLSX)
```

---

## 6. n8n Environment Variables

Set in n8n **Settings → Environment Variables**:
```env
# Core
BASE_URL=https://your-tunnel.trycloudflare.com

# Telegram
TG_CHAT_ID=123456789

# LLM Provider (gemini / openai)
LLM_PROVIDER=gemini

# SMTP (для Email-отчётов)
SMTP_FROM=noreply@sfedu.ru
REPORT_EMAIL=admin@sfedu.ru
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

---

## 9. Workflow Validation Checklist

Before activating ANY workflow, verify:

### Implementation
- [ ] Trigger configured (Schedule / Webhook / Manual)
- [ ] All HTTP Request nodes use Generic Credential (Bearer N8N_API_KEY)
- [ ] Webhook data accessed via `$json.body.field` (not `$json.field`)
- [ ] At least 1 data transformation node (Set / Code / IF)
- [ ] Output/action nodes present (Telegram / API / Email)

### Error Handling
- [ ] **Error Trigger** node added (catches all unhandled errors)
- [ ] Error notification path (Telegram or log)
- [ ] Unary operators use `isNotEmpty` / `isEmpty` (not manual singleValue)
- [ ] Branch `false` output connected or intentionally empty

### Data Validation
- [ ] IF node validates critical inputs before processing
- [ ] Empty results handled gracefully (IF with no action = silent skip)
- [ ] `.get()` or `||` used in Code nodes for safe access
- [ ] Return format: always `[{"json": {…}}]`

### Weekly Report Specific
- [ ] **Switch LLM** correctly routes to Gemini / OpenAI по `$env.LLM_PROVIDER`
- [ ] **Merge Data Sources** has `combineByPosition` mode with 5 inputs
- [ ] **Postgres** table `weekly_reports` exists (schema ниже)
- [ ] **Error Workflow** ID указан в `settings.errorWorkflow`
- [ ] All HTTP Request nodes use `$env.BASE_URL` (не хардкод)
- [ ] Telegram message uses HTML parse_mode (не Markdown)
- [ ] LLM prompt запрашивает JSON с ключами: `executive_summary`, `top_skills_analysis`, `key_trends`, `profile_highlights`, `recommendations`, `market_insights`
- [ ] Parse Report имеет fallback при невалидном JSON от LLM

---

## 10. Deployment Checklist

- [ ] `.env` has `N8N_API_KEY` set
- [ ] `.env` has `N8N_WEBHOOK_SECRET` set
- [ ] API is reachable from n8n (test: `curl /health`)
- [ ] Generic Credential created in n8n with `Authorization: Bearer <N8N_API_KEY>`
- [ ] X-N8N-Webhook-Secret matches in n8n HTTP Request headers
- [ ] Workflow imported from `src/n8n/workflows/*.json`
- [ ] Workflow validated (no red nodes)
- [ ] Tested with manual trigger before scheduling
- [ ] Monitoring: check first 3 automated executions

### Weekly Report Deployment
- [ ] **Postgres** таблица создана (см. `n8n.py` или выполнить SQL ниже)
- [ ] **LLM credentials** настроены: Google Palm API + OpenAI API
- [ ] **SMTP credential** создан в n8n: `SFEDU SMTP` с вашими SMTP-настройками
- [ ] **Error Workflow** ID вставлен в `settings.errorWorkflow`
- [ ] **Environment variables** установлены:
  ```env
  BASE_URL=https://your-tunnel.trycloudflare.com
  TG_CHAT_ID=<chat_id>
  LLM_PROVIDER=gemini
  SMTP_FROM=noreply@sfedu.ru
  REPORT_EMAIL=admin@sfedu.ru
  ```
- [ ] **Schedule** активен: Monday 09:00
- [ ] **Cloudflare tunnel** запущен: `cloudflared tunnel --url http://localhost:8000`

### PostgreSQL Table Schema
```sql
CREATE TABLE IF NOT EXISTS weekly_reports (
    id SERIAL PRIMARY KEY,
    executive_summary TEXT,
    report_json JSONB,
    skill_count INT,
    profile_count INT,
    generated_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```
