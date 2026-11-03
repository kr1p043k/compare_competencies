import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../api";
import { PublicationsTab } from "./PublicationsTab";
import type { Publication } from "./publicationsFilter";

vi.mock("../api", () => ({ api: vi.fn() }));

// Явный cleanup: при singleFork автоочистка RTL не срабатывает между файлами,
// и разметка предыдущего теста остаётся в document (ломает поиск по роли).
afterEach(cleanup);

const apiMock = vi.mocked(api);

function makePub(id: number, title: string, extra: Partial<Publication> = {}): Publication {
  return {
    id,
    title,
    authors: [`Автор ${id}`],
    type_id: 39,
    type_name: "Научная статья",
    signature: null,
    doi: null,
    isbn: null,
    track_id: null,
    created_at: null,
    year: 2025,
    academic_subject: [],
    education_program: [],
    abstract: null,
    keywords: [],
    journal: null,
    publisher: null,
    language: null,
    publishing_year: null,
    pages: null,
    files: [],
    url: null,
    web_url: `https://hub.sfedu.ru/dashboard/asset/${id}/`,
    creator_username: null,
    ...extra,
  };
}

const PAGE_ONE = [
  makePub(1, "Моделирование графов", { abstract: "Алгоритмы обхода графа" }),
  makePub(2, "Нейронные сети в образовании"),
  makePub(3, "Экономика региона"),
];

const TYPES = {
  types: [
    { id: 39, name: "Научная статья", count: 21591 },
    { id: 34, name: "Выпускная квалификационная работа", count: 61807 },
  ],
  exact: true,
  total: 133902,
};

function route(endpoint: string) {
  if (endpoint.startsWith("/publications/types")) return Promise.resolve(TYPES);
  if (endpoint.startsWith("/publications?")) {
    return Promise.resolve({ items: PAGE_ONE, total: 133902, page: 1, page_size: 20, num_pages: 6696 });
  }
  return Promise.reject(new Error(`unexpected endpoint: ${endpoint}`));
}

beforeEach(() => {
  apiMock.mockReset();
  apiMock.mockImplementation(route as never);
});

describe("PublicationsTab", () => {
  it("показывает честный счётчик загруженного", async () => {
    render(<PublicationsTab />);
    expect(await screen.findByText(/Загружено 3 из 133 902/)).toBeInTheDocument();
  });

  it("показывает тип с точным счётчиком в подписи", async () => {
    render(<PublicationsTab />);
    await screen.findByText(/Загружено 3 из/);
    expect(screen.getByRole("combobox", { name: "Фильтр по типу публикации" })).toBeInTheDocument();
    expect(screen.getByText("Все типы")).toBeInTheDocument();
  });

  it("фильтрует загруженные записи локально, не дергая сервер", async () => {
    const user = userEvent.setup();
    render(<PublicationsTab />);
    await screen.findByText("Моделирование графов");

    const input = screen.getByPlaceholderText("Фильтр по названию, авторам, журналу, аннотации");
    await user.type(input, "граф");

    await waitFor(() => expect(screen.queryByText("Экономика региона")).not.toBeInTheDocument());
    expect(screen.getByText("Моделирование графов")).toBeInTheDocument();
    expect(screen.getByText(/Совпадений: 1 среди 3 загруженных/)).toBeInTheDocument();

    const calledSearch = apiMock.mock.calls.some(([endpoint]) => String(endpoint).includes("/publications/search"));
    expect(calledSearch).toBe(false);
  });

  it("честно сообщает, когда совпадений нет среди загруженных", async () => {
    const user = userEvent.setup();
    render(<PublicationsTab />);
    await screen.findByText("Моделирование графов");

    await user.type(screen.getByPlaceholderText("Фильтр по названию, авторам, журналу, аннотации"), "квантовая хромодинамика");

    expect(await screen.findByText("Среди загруженных записей совпадений нет")).toBeInTheDocument();
    expect(screen.getByText(/Просмотрено 3 из 133 902 записей/)).toBeInTheDocument();
  });

  it("догружает следующую страницу кнопкой", async () => {
    const user = userEvent.setup();
    apiMock.mockImplementation(((endpoint: string) => {
      if (endpoint.startsWith("/publications/types")) return Promise.resolve(TYPES);
      if (endpoint.includes("page=2")) {
        return Promise.resolve({ items: [makePub(4, "Четвёртая запись")], total: 133902, page: 2, page_size: 20, num_pages: 6696 });
      }
      return Promise.resolve({ items: PAGE_ONE, total: 133902, page: 1, page_size: 20, num_pages: 6696 });
    }) as never);

    render(<PublicationsTab />);
    await screen.findByText("Моделирование графов");

    await user.click(screen.getAllByRole("button", { name: /Загрузить ещё/ })[0]);

    expect(await screen.findByText("Четвёртая запись")).toBeInTheDocument();
    expect(screen.getByText(/Загружено 4 из 133 902/)).toBeInTheDocument();
  });

  it("ведёт ссылку на карточку, а не на API", async () => {
    render(<PublicationsTab />);
    await screen.findByText("Моделирование графов");
    const link = screen.getAllByRole("link", { name: /Открыть в хабе/ })[0];
    expect(link).toHaveAttribute("href", "https://hub.sfedu.ru/dashboard/asset/1/");
  });
});
