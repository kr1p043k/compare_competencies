import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { request } from "../../lib/api";
import { useApi } from "../../lib/useApi";
import { AnalysisTab } from "./AnalysisTab";

vi.mock("../../lib/useApi", () => ({ useApi: vi.fn() }));
vi.mock("../../lib/api", () => ({ request: vi.fn() }));

afterEach(cleanup);

const useApiMock = vi.mocked(useApi);
const requestMock = vi.mocked(request);

function query<T>(data: T | null) {
  return { data, error: null, loading: false, refetch: () => {} };
}

describe("AnalysisTab", () => {
  beforeEach(() => {
    useApiMock.mockReset();
    requestMock.mockReset();
  });

  it("renders the header and requests the professions list", () => {
    useApiMock.mockReturnValue(query({ professions: [{ name: "Backend" }] }) as never);
    render(<AnalysisTab selectedProfile="" onProfileChange={() => {}} />);

    expect(screen.getByRole("combobox")).toBeTruthy();
    expect(useApiMock).toHaveBeenCalledWith("/api/trends/professions");
  });

  it("shows pipeline badges for the query and the regions", () => {
    useApiMock.mockReturnValue(query({ professions: [] }) as never);
    const { container, rerender } = render(
      <AnalysisTab selectedProfile="" onProfileChange={() => {}} pipelineQuery="python" pipelineRegions="1,2,3" />,
    );
    expect(container.textContent).toContain("python");
    expect(container.textContent).toContain("3");

    rerender(<AnalysisTab selectedProfile="" onProfileChange={() => {}} pipelineRegions="0" />);
    expect(container.textContent).not.toContain("python");
  });

  it("loads skills for the picked profession and shows frequency", async () => {
    useApiMock.mockReturnValue(query({ professions: [{ name: "Backend" }] }) as never);
    requestMock.mockResolvedValue({ skills: [{ skill: "SQL", frequency: 42 }] });
    const user = userEvent.setup();
    render(<AnalysisTab selectedProfile="" onProfileChange={() => {}} />);

    await user.click(screen.getByRole("combobox"));
    await user.click(await screen.findByRole("option", { name: "Backend" }));

    expect(await screen.findByText("SQL")).toBeTruthy();
    expect(screen.getByText(/42/)).toBeTruthy();
    expect(requestMock).toHaveBeenCalledWith(
      "/api/trends/by-profession?profession=Backend&limit=30",
    );
  });

  it("shows the error message when the request fails", async () => {
    useApiMock.mockReturnValue(query({ professions: [{ name: "Backend" }] }) as never);
    requestMock.mockRejectedValue(new Error("network down"));
    const user = userEvent.setup();
    render(<AnalysisTab selectedProfile="" onProfileChange={() => {}} />);

    await user.click(screen.getByRole("combobox"));
    await user.click(await screen.findByRole("option", { name: "Backend" }));

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("network down");
  });

  it("shows an empty state when a profession has no skills", async () => {
    useApiMock.mockReturnValue(query({ professions: [{ name: "Backend" }] }) as never);
    requestMock.mockResolvedValue({ skills: [] });
    const user = userEvent.setup();
    render(<AnalysisTab selectedProfile="" onProfileChange={() => {}} />);

    await user.click(screen.getByRole("combobox"));
    await user.click(await screen.findByRole("option", { name: "Backend" }));

    expect(await screen.findByText(/Нет данных/)).toBeTruthy();
  });
});