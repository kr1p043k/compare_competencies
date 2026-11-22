/**
 * a11y: MonitoringTab без нарушений axe после загрузки всех блоков
 * (метрики, здоровье, Grafana, подписки, уведомления).
 *
 * Сеть мокается на уровне lib/api.request по образцу
 * MonitoringTab.coverage.test.tsx; fetch подменён заглушкой, чтобы
 * reportError не ходил в /api/log.
 *
 * Без установленного jest-axe тест падает с подсказкой по установке,
 * а не пропускается (см. src/test/a11y.ts).
 */
import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, it, vi } from "vitest";

import { request } from "../../lib/api";
import { expectNoA11yViolations } from "../../test/a11y";
import { MonitoringTab } from "./MonitoringTab";

vi.mock("../../lib/api", () => {
  class ApiError extends Error {
    constructor(
      message: string,
      readonly status = 0,
      readonly detail: unknown = null,
      readonly url = "",
    ) {
      super(message);
      this.name = "ApiError";
    }
  }
  return { request: vi.fn(), ApiError };
});

const requestMock = vi.mocked(request);

const METRICS = {
  pipeline_stage_duration_seconds: {
    name: "pipeline_stage_duration_seconds",
    type: "histogram",
    samples: [
      {
        name: "pipeline_stage_duration_seconds_count",
        labels: { stage: "data_collection" },
        value: 3,
      },
      {
        name: "pipeline_stage_duration_seconds_sum",
        labels: { stage: "data_collection" },
        value: 9,
      },
    ],
  },
  pipeline_errors: {
    name: "pipeline_errors",
    type: "counter",
    samples: [
      { name: "pipeline_errors_total", labels: { stage: "data_collection" }, value: 2 },
    ],
  },
  ltr_model_metric: {
    name: "ltr_model_metric",
    type: "gauge",
    samples: [{ name: "ltr_model_metric", labels: { metric: "ndcg" }, value: 0.8123 }],
  },
  recommendations_generated: {
    name: "recommendations_generated",
    type: "counter",
    samples: [{ name: "recommendations_generated_total", labels: {}, value: 42 }],
  },
  api_requests: {
    name: "api_requests",
    type: "counter",
    samples: [{ name: "api_requests_total", labels: {}, value: 1234 }],
  },
};

const HEALTH = {
  sessions_active: 5,
  errors_1h: { "4xx": 2, "5xx": 1 },
  latency_p95_by_endpoint: { "/api/vacancies": 0.123, "/api/reports": null },
  recent_runs: [
    { action: "parse", status: "completed", started_at: "2026-11-18T10:00:00Z" },
    { action: "train", status: "failed", started_at: null },
  ],
  freshness: { vacancies: 12345, date_from: "2026-11-01", date_to: "2026-11-18" },
};

const GRAFANA = {
  enabled: true,
  reachable: true,
  dashboards: [
    { uid: "g1", title: "Pipelines", url: "http://grafana.local/d/pipelines" },
    { uid: "g2", title: "Без ссылки", url: null },
  ],
};

const SUB = {
  id: "s1",
  topic: "LLM",
  source: "arxiv",
  telegram_chat_id: "123456",
  email: null,
  is_active: true,
  last_checked_at: "2026-11-18T10:00:00Z",
  created_at: "2026-11-01T00:00:00Z",
};

const NOTIF = {
  id: "n1",
  subscription_id: "s1",
  title: "Новая статья",
  body: "Про LLM и графы",
  article_url: "https://example.org/article",
  article_source: "OpenAlex",
  severity: "error",
  is_read: false,
  created_at: "2026-11-18T09:00:00Z",
};

afterEach(() => {
  vi.unstubAllGlobals();
});

beforeEach(() => {
  requestMock.mockReset();
  vi.stubGlobal(
    "fetch",
    vi.fn(() => Promise.resolve(new Response("{}", { status: 200 }))),
  );
  vi.spyOn(console, "error").mockImplementation(() => {});

  requestMock.mockImplementation(((url: string) => {
    const u = String(url);
    if (u === "/api/admin/monitoring") return Promise.resolve(METRICS);
    if (u === "/api/admin/monitoring/summary") return Promise.resolve(HEALTH);
    if (u === "/api/admin/monitoring/grafana") return Promise.resolve(GRAFANA);
    if (u === "/api/subscriptions") return Promise.resolve({ subscriptions: [SUB] });
    if (u.startsWith("/api/notifications/unread-count")) {
      return Promise.resolve({ unread_count: 1 });
    }
    if (u.startsWith("/api/notifications")) {
      return Promise.resolve({ notifications: [NOTIF] });
    }
    return Promise.reject(new Error(`unexpected endpoint: ${u}`));
  }) as never);
});

describe("MonitoringTab (a11y)", () => {
  it("после загрузки всех блоков - без нарушений", async () => {
    const { container } = render(<MonitoringTab />);

    await screen.findByText("Запуски пайплайна");
    await screen.findByText("Новая статья");

    await expectNoA11yViolations(container);
  });
});