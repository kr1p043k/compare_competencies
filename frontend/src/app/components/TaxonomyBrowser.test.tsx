/**
 * TaxonomyBrowser: загрузка категорий, поиск, пусто/ошибка (7.5.4).
 * Сеть мокается на уровне app/api (api()).
 */
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { api } from "../api";
import { TaxonomyBrowser } from "./TaxonomyBrowser";

vi.mock("../api", () => ({ api: vi.fn() }));
vi.mock("./../../lib/errors", () => ({ reportError: vi.fn() }));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

const apiMock = vi.mocked(api);
const SEARCH = "Найти навык или категорию...";

const CATS = {
  categories: [
    { id: "lang", label: "Языки", icon: "code", total: 2, skills: ["python", "go"] },
    { id: "db", label: "Базы данных", icon: "db", total: 1, skills: ["sql"] },
  ],
};

describe("TaxonomyBrowser", () => {
  it("рендерит заголовок и категории", async () => {
    apiMock.mockResolvedValue(CATS as never);
    render(<TaxonomyBrowser />);
    expect(screen.getAllByText("Таксономия навыков").length).toBeGreaterThan(0);
    await waitFor(() => expect(screen.getByText("Языки")).toBeTruthy());
    expect(screen.getByText("Базы данных")).toBeTruthy();
  });

  it("ошибка загрузки -> ErrorState", async () => {
    apiMock.mockRejectedValue(new Error("нет связи"));
    render(<TaxonomyBrowser />);
    await waitFor(() => expect(screen.getByText(/нет связи/)).toBeTruthy());
  });

  it("пустой ответ -> EmptyState", async () => {
    apiMock.mockResolvedValue({ categories: [] } as never);
    render(<TaxonomyBrowser />);
    await waitFor(() => expect(screen.getByText("Ничего не найдено")).toBeTruthy());
  });

  it("поиск по навыку фильтрует категории", async () => {
    apiMock.mockResolvedValue(CATS as never);
    render(<TaxonomyBrowser />);
    await waitFor(() => expect(screen.getByText("Базы данных")).toBeTruthy());
    fireEvent.change(screen.getByPlaceholderText(SEARCH), { target: { value: "python" } });
    await waitFor(() => expect(screen.queryByText("Базы данных")).toBeNull());
    expect(screen.getByText("Языки")).toBeTruthy();
  });
});
