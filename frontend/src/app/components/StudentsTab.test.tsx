/**
 * StudentsTab: список студентов, пусто/ошибка, открытие детали (7.5.4).
 * Сеть мокается на уровне app/api (api()).
 */
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { api } from "../api";
import { StudentsTab } from "./StudentsTab";

vi.mock("../api", () => ({ api: vi.fn() }));
vi.mock("../../lib/errors", () => ({
  errMessage: (e: unknown) => (e instanceof Error ? e.message : String(e ?? "")),
}));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

const apiMock = vi.mocked(api);

const ROW = {
  email: "s@sfedu.ru",
  full_name: "Иван Студентов",
  target_level: "middle",
  skills_count: 7,
  competencies_count: 3,
  has_profile: true,
};

describe("StudentsTab", () => {
  it("рендерит список студентов", async () => {
    apiMock.mockResolvedValue({ students: [ROW] } as never);
    render(<StudentsTab />);
    expect(screen.getByText("Студенты")).toBeTruthy();
    await waitFor(() => expect(screen.getByText("Иван Студентов")).toBeTruthy());
    expect(screen.getByText(/s@sfedu\.ru/)).toBeTruthy();
  });

  it("пустой список -> EmptyState", async () => {
    apiMock.mockResolvedValue({ students: [] } as never);
    render(<StudentsTab />);
    await waitFor(() =>
      expect(screen.getByText("Студентов с ролью student пока нет")).toBeTruthy(),
    );
  });

  it("ошибка загрузки -> ErrorState", async () => {
    apiMock.mockRejectedValue(new Error("нет доступа"));
    render(<StudentsTab />);
    await waitFor(() => expect(screen.getByText(/нет доступа|Нет доступа/)).toBeTruthy());
  });

  it("клик по студенту грузит детали", async () => {
    apiMock.mockImplementation(async (url: string) => {
      if (url.startsWith("/teacher/students")) return { students: [ROW] } as never;
      return {
        email: ROW.email,
        full_name: ROW.full_name,
        target_level: "middle",
        skills: ["python", "sql"],
        user_added: ["docker"],
        competencies: ["ПК-1"],
        updated_at: null,
        has_profile: true,
      } as never;
    });
    render(<StudentsTab />);
    await waitFor(() => expect(screen.getByText("Иван Студентов")).toBeTruthy());
    fireEvent.click(screen.getByText("Иван Студентов"));
    await waitFor(() =>
      expect(apiMock).toHaveBeenCalledWith(expect.stringContaining("/admin/students/skills?email=")),
    );
    await waitFor(() => expect(screen.getByText("python")).toBeTruthy());
  });
});