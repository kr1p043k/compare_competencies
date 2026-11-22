import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { request } from "../../lib/api";
import { useAuth } from "../../lib/auth";
import { ScientificTrendsTab } from "./ScientificTrendsTab";

vi.mock("../../lib/api", () => ({ request: vi.fn() }));
vi.mock("../../lib/auth", () => ({ useAuth: vi.fn() }));
vi.mock("../../lib/errors", () => ({ reportError: vi.fn() }));

afterEach(cleanup);

const requestMock = vi.mocked(request);
const useAuthMock = vi.mocked(useAuth);

function okResponse(payload: unknown) {
  return {
    ok: true,
    status: 200,
    json: async () => payload,
  } as unknown as Response;
}

function failResponse(status: number, detail: string) {
  return {
    ok: false,
    status,
    json: async () => ({ detail }),
  } as unknown as Response;
}

const TREND = {
  topic: "ml",
  found_trends: ["LLM"],
  recommended_competencies: [
    {
      code: "PK-1",
      description: "Machine learning",
      keywords: "ml,ai",
      trend_source: "market",
      coverage_percent: 50,
      skills_count: 4,
    },
  ],
  rationale: "Because",
};

describe("ScientificTrendsTab", () => {
  beforeEach(() => {
    requestMock.mockReset();
    useAuthMock.mockReturnValue({ role: "teacher", name: "T", username: "t", authed: true } as never);
  });

  it("renders the page header and both actions for a teacher", () => {
    render(<ScientificTrendsTab />);
    expect(screen.getByText(/Научные тренды/)).toBeTruthy();
    const gap = screen.getByRole("button", { name: /разрыва/i });
    expect((gap as HTMLButtonElement).disabled).toBe(false);
  });

  it("disables gap analysis for students", () => {
    useAuthMock.mockReturnValue({ role: "student", name: "S", username: "s", authed: true } as never);
    render(<ScientificTrendsTab />);
    const gap = screen.getByRole("button", { name: /разрыва/i });
    expect((gap as HTMLButtonElement).disabled).toBe(true);
  });

  it("asks for a topic before calling the API", async () => {
    const user = userEvent.setup();
    render(<ScientificTrendsTab />);
    await user.click(screen.getByRole("button", { name: /компетенции/i }));
    expect(await screen.findByRole("alert")).toBeTruthy();
    expect(requestMock).not.toHaveBeenCalled();
  });

  it("loads recommended competencies for a topic", async () => {
    requestMock.mockResolvedValue(okResponse(TREND));
    const user = userEvent.setup();
    render(<ScientificTrendsTab />);

    await user.type(screen.getByRole("textbox"), "ml");
    await user.click(screen.getByRole("button", { name: /компетенции/i }));

    expect(await screen.findByText("PK-1")).toBeTruthy();
    expect(requestMock).toHaveBeenCalledWith(
      "/api/academic/get-competencies",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("shows the backend error detail when the call fails", async () => {
    requestMock.mockResolvedValue(failResponse(502, "hub is down"));
    const user = userEvent.setup();
    render(<ScientificTrendsTab />);

    await user.type(screen.getByRole("textbox"), "ml");
    await user.click(screen.getByRole("button", { name: /компетенции/i }));

    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("hub is down"));
  });
});