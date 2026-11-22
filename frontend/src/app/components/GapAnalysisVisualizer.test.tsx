import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../api";
import { GapAnalysisVisualizer } from "./GapAnalysisVisualizer";

vi.mock("../api", () => ({ api: vi.fn() }));
vi.mock("../../lib/errors", () => ({ reportError: vi.fn() }));
vi.mock("./ShapExplanationCard", () => ({
  ShapExplanationCard: () => <div data-testid="shap-stub" />,
}));

afterEach(cleanup);

const apiMock = vi.mocked(api);

describe("GapAnalysisVisualizer", () => {
  beforeEach(() => {
    apiMock.mockReset();
  });

  it("requests recommendations for the active profile", async () => {
    apiMock.mockResolvedValue({
      recommendations: [{ skill: "python", importance_score: 0.82 }],
    });
    render(<GapAnalysisVisualizer profile="base" />);

    expect(screen.getByText("Визуализация GAP-анализа")).toBeTruthy();
    await waitFor(() => expect(apiMock).toHaveBeenCalledWith("/results/recommendations/base"));
    expect(await screen.findByTestId("shap-stub")).toBeTruthy();
    expect(screen.getByText(/ТОП важных навыков/)).toBeTruthy();
  });

  it("shows an empty state when there are no recommendations", async () => {
    apiMock.mockResolvedValue({ recommendations: [], message: "Модель не обучена" });
    render(<GapAnalysisVisualizer profile="dc" />);

    expect(await screen.findByText(/Нет ML-рекомендаций/)).toBeTruthy();
    expect(screen.getByText("Модель не обучена")).toBeTruthy();
  });

  it("shows an error state and retries", async () => {
    apiMock.mockRejectedValueOnce(new Error("boom")).mockResolvedValueOnce({
      recommendations: [{ skill: "sql", importance_score: 0.5 }],
    });
    render(<GapAnalysisVisualizer profile="top_dc" />);

    const retry = await screen.findByRole("button", { name: /Повторить/ });
    await userEvent.click(retry);

    await waitFor(() => expect(apiMock).toHaveBeenCalledTimes(2));
    expect(await screen.findByText(/ТОП важных навыков/)).toBeTruthy();
  });

  it("switches to cluster insights and reports a missing image", async () => {
    apiMock.mockResolvedValue({ recommendations: [] });
    render(<GapAnalysisVisualizer profile="base" />);

    await userEvent.click(screen.getByRole("button", { name: /Кластерные инсайты/ }));

    const img = document.querySelector('img[src="/api/results/images/base/cluster_insights"]');
    expect(img).not.toBeNull();

    fireEvent.error(img!);
    expect(await screen.findByText("Визуализация недоступна")).toBeTruthy();
  });

  it("notifies the parent when the profile changes", async () => {
    apiMock.mockResolvedValue({ recommendations: [] });
    const onProfileChange = vi.fn();
    render(<GapAnalysisVisualizer profile="base" onProfileChange={onProfileChange} />);

    await userEvent.click(screen.getByRole("combobox"));
    await userEvent.click(await screen.findByRole("option", { name: /Data Scientist/ }));

    expect(onProfileChange).toHaveBeenCalledWith("dc");
  });
});