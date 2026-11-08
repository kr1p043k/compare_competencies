/**
 * VacanciesList: CTA «Собрать вакансии» в сводной карточке сбора.
 * Сеть мокается на уровне lib/api (request), fetch не вызывается.
 *
 * Регрессия «кнопка сливается с окружением»:
 *  - акцентные цвета обязаны приходить из дизайн-токенов (bg-primary =
 *    var(--primary), светлая #2563eb / тёмная #3b82f6), а не из хардкода
 *    bg-blue-700/800: в тёмной теме #1d4ed8 поверх карточки #111a2c даёт
 *    2.59:1 (< 3:1 WCAG 1.4.11) и кнопка растворяется в фоне;
 *  - px-6 обязан перекрывать служебный has-[>svg]:px-2.5 из size="sm"
 *    (tw-merge конфликтует только одноимённые варианты, а в собранном CSS
 *    :has(>svg) выигрывает по специфичности - фактический паддинг
 *    кнопки с иконкой схлопывался до 10px вместо 24px);
 *  - на узких экранах кнопка растягивается на колонку (w-full md:w-auto),
 *    строка умеет переноситься (flex-wrap) и не жмёт текст.
 */
import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { request } from "../../lib/api";
import { VacanciesList } from "./VacanciesList";

vi.mock("../../lib/api", () => ({
  download: vi.fn(),
  request: vi.fn(),
  saveBlob: vi.fn(),
}));

afterEach(cleanup);

const requestMock = vi.mocked(request);

const ITEMS = [1, 2].map((id) => ({
  id: `v-${id}`,
  name: `Вакансия ${id}`,
  experience: "От 1 года до 3 лет",
  salary_from: 100000,
  salary_to: 150000,
  salary_currency: "RUR",
  employer_name: "Компания",
  area: "Москва",
  published_at: "2026-10-01T00:00:00",
  alternate_url: `https://hh.ru/vacancy/${id}`,
  skills: ["Python", "SQL"],
  snippet: { requirement: "Python", responsibility: "Разработка" },
}));

const INFO = {
  count: 2,
  with_skills: 2,
  file_modified: "2026-11-01 12:00",
  date_range: { from: "2026-10-01", to: "2026-11-01" },
  load_error: null,
};

function route(path: string) {
  if (path.startsWith("/api/vacancies/info")) return Promise.resolve(INFO);
  if (path.startsWith("/api/vacancies?")) {
    return Promise.resolve({ items: ITEMS, total: ITEMS.length, limit: 12, offset: 0, has_more: false });
  }
  return Promise.reject(new Error(`unexpected endpoint: ${path}`));
}

beforeEach(() => {
  requestMock.mockReset();
  requestMock.mockImplementation(route as never);
});

function classList(cls: string): string[] {
  return cls.split(/\s+/).filter(Boolean);
}

describe("VacanciesList: CTA «Собрать вакансии»", () => {
  it("берёт акцент из дизайн-токенов (bg-primary), а не хардкод blue-700/800", async () => {
    render(<VacanciesList />);
    const btn = await screen.findByRole("button", { name: "Собрать вакансии" });
    const cls = classList(btn.className);
    expect(cls).toContain("bg-primary");
    expect(cls).toContain("text-primary-foreground");
    expect(cls).toContain("hover:bg-primary/90");
    expect(cls).not.toContain("bg-blue-700");
    expect(cls).not.toContain("hover:bg-blue-800");
    expect(cls).not.toContain("text-white");
  });

  it("не даёт паддингам схлопнуться до has-[>svg]:px-2.5 от size=\"sm\"", async () => {
    render(<VacanciesList />);
    const btn = await screen.findByRole("button", { name: "Собрать вакансии" });
    const cls = classList(btn.className);
    expect(cls).toContain("px-6");
    expect(cls).toContain("has-[>svg]:px-6");
    expect(cls).not.toContain("has-[>svg]:px-2.5");
    expect(cls).not.toContain("px-3");
  });

  it("на узких экранах растягивается на колонку, от md - авто (w-full md:w-auto)", async () => {
    render(<VacanciesList />);
    const btn = await screen.findByRole("button", { name: "Собрать вакансии" });
    const cls = classList(btn.className);
    expect(cls).toContain("w-full");
    expect(cls).toContain("md:w-auto");
    // иконка не добавляет mr- поверх gap-класса кнопки
    expect(btn.querySelector("svg")?.getAttribute("class")).not.toContain("mr-1");
  });

  it("строка CTA переносится при нехватке места (flex-wrap), текст может сжиматься (min-w-0)", async () => {
    render(<VacanciesList />);
    const btn = await screen.findByRole("button", { name: "Собрать вакансии" });
    const row = classList(btn.parentElement?.className ?? "");
    expect(row).toContain("md:flex-wrap");
    expect(row).toContain("gap-3");
  });

  it("клик открывает панель настроек сбора - поведение не тронуто", async () => {
    const user = userEvent.setup();
    render(<VacanciesList />);
    const btn = await screen.findByRole("button", { name: "Собрать вакансии" });
    await user.click(btn);
    expect(await screen.findByRole("button", { name: "Запустить сбор" })).toBeInTheDocument();
    expect(screen.getByText("Настройки сбора вакансий")).toBeInTheDocument();
  });
});

describe("VacanciesList: панель сбора и «Поиск по всему рынку»", () => {
  /** Открыть панель настроек сбора, затем блок списка городов. */
  async function openCityPicker(): Promise<HTMLElement> {
    const user = userEvent.setup();
    render(<VacanciesList />);
    const cta = await screen.findByRole("button", { name: "Собрать вакансии" });
    await user.click(cta);
    const toggler = await screen.findByRole("button", {
      name: "Показать список городов и вакансий для сбора",
    });
    await user.click(toggler);
    return screen.getByRole("group", { name: "Города для сбора" });
  }

  it("«Запустить сбор» - CTA на токенах, читаем в обеих темах", async () => {
    const user = userEvent.setup();
    render(<VacanciesList />);
    await user.click(await screen.findByRole("button", { name: "Собрать вакансии" }));
    const run = await screen.findByRole("button", { name: "Запустить сбор" });
    const cls = classList(run.className);
    // явные акцентные токены вместо bg-card/bg-muted
    expect(cls).toContain("bg-primary");
    expect(cls).toContain("text-primary-foreground");
    expect(cls).toContain("hover:bg-primary/90");
    // старый вид: плашка цвета карточки + текст primary-foreground (белый на белом)
    expect(cls).not.toContain("bg-card");
    expect(cls).not.toContain("hover:bg-muted");
    expect(cls).not.toContain("dark:bg-muted");
    expect(cls).not.toContain("dark:hover:bg-slate-200");
  });

  it("«Поиск по всему рынку» без счётчиков базы", async () => {
    const group = await openCityPicker();
    expect(group).toBeInTheDocument();
    // счётчик «(в базе: 2 шт., с навыками: 2 шт.)» убран из текста блока
    expect(screen.queryByText(/в базе:/)).not.toBeInTheDocument();
    expect(screen.queryByText(/с навыками:/)).not.toBeInTheDocument();
    const p = screen.getByText(/Будут собраны вакансии/);
    expect(p.textContent).not.toMatch(/шт\./);
  });

  it("«Поиск по всему рынку» на нейтральных токенах, без голубой обёртки", async () => {
    await openCityPicker();
    const header = screen.getByText("Поиск по всему рынку");
    const wrap = classList(header.parentElement?.className ?? "");
    expect(wrap).toContain("bg-muted");
    expect(wrap).toContain("border-border");
    // голубой хардкод обёртки удалён
    expect(wrap).not.toContain("bg-primary-soft");
    expect(wrap).not.toContain("dark:bg-blue-950/20");
    expect(wrap).not.toContain("border-primary/30");
    expect(classList(header.className)).not.toContain("text-blue-800");
  });

  it("города: chip-сетка из дизайн-системы, без букв и сырых счётчиков", async () => {
    const group = await openCityPicker();
    // одна плоская сетка кнопок-городов: ровно 100 чипов HH_REGIONS,
    // букв-заголовков А/Б/В и плашек «n/m» больше нет
    const chips = within(group).getAllByRole("button");
    expect(chips).toHaveLength(100);
    for (const chip of chips) {
      expect(chip.getAttribute("aria-pressed")).toMatch(/^(true|false)$/);
    }
    expect(within(group).queryByText(/^\d+\/\d+$/)).not.toBeInTheDocument();
  });

  it("города: поиск «Найти город...» и мультивыбор сохранены", async () => {
    const group = await openCityPicker();
    const search = screen.getByPlaceholderText("Найти город...");
    // мультивыбор: два города независимо переключаются
    const moscow = within(group).getByRole("button", { name: "Москва" });
    expect(moscow.getAttribute("aria-pressed")).toBe("false");
    const user = userEvent.setup();
    await user.click(moscow);
    expect(moscow.getAttribute("aria-pressed")).toBe("true");
    expect(await screen.findByText("выбрано: 1")).toBeInTheDocument();
    const ufa = within(group).getByRole("button", { name: "Уфа" });
    await user.click(ufa);
    expect(ufa.getAttribute("aria-pressed")).toBe("true");
    expect(await screen.findByText("выбрано: 2")).toBeInTheDocument();
    // повторный клик снимает выбор
    await user.click(moscow);
    expect(moscow.getAttribute("aria-pressed")).toBe("false");
    expect(await screen.findByText("выбрано: 1")).toBeInTheDocument();
    // поиск фильтрует сетку, не сбрасывая выборку
    await user.type(search, "Моск");
    expect(within(group).queryByRole("button", { name: "Уфа" })).not.toBeInTheDocument();
    const filtered = within(group).getByRole("button", { name: "Москва" });
    expect(filtered).toBeInTheDocument();
    expect(filtered.getAttribute("aria-pressed")).toBe("false");
    expect(await screen.findByText("выбрано: 1")).toBeInTheDocument();
  });
});
