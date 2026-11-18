/**
 * StudentKrm: программа КРМ глазами студента - загрузка направлений,
 * компактный список дисциплин, фильтры, оптимистичное добавление навыков.
 * Сеть мокается на уровне lib/api.request, fetch подменён заглушкой
 * (reportError пишет в /api/log - реальных запросов быть не должно).
 */
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { request } from "../../lib/api";
import { StudentKrm } from "./StudentKrm";

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

const DIRECTIONS = {
  directions: [{ dir_code: "09.03.02", name: "Информатика и вычислительная техника" }],
};

const PROFILES = { profiles: ["base", "dc", "top_dc"] };

const KRM = {
  direction: "09.03.02",
  direction_name: "Информатика и вычислительная техника",
  profile: "base",
  disciplines: [
    {
      discipline: "Базы данных",
      competencies: [
        {
          code: "ПК-1",
          skills: [
            { text: "SQL", student_has: true },
            { text: "Индексы", student_has: false },
          ],
        },
      ],
    },
    {
      discipline: "Сети",
      competencies: [{ code: "ПК-2", skills: [] }],
    },
  ],
  merged: [],
  counts: { krm: 2, student: 1, overlap: 1, merged: 0, student_only: 0 },
};

const EMPTY_KRM = {
  ...KRM,
  disciplines: [],
  counts: { krm: 0, student: 0, overlap: 0, merged: 0, student_only: 0 },
};

type Route = (url: string, options?: { method?: string; body?: unknown }) => Promise<unknown>;

function makeRoute(overrides: Partial<Record<string, unknown>> = {}): Route {
  return (url, options) => {
    if (url === "/api/krm/directions") return Promise.resolve(overrides.directions ?? DIRECTIONS);
    if (url === "/api/profiles") return Promise.resolve(overrides.profiles ?? PROFILES);
    if (url.startsWith("/api/krm/student/competencies")) {
      return Promise.resolve(overrides.krm ?? KRM);
    }
    if (url === "/api/profiles/self" && options?.method === "PATCH") {
      if (overrides.patchRejects) {
        return Promise.reject(new Error("Сервер недоступен"));
      }
      return Promise.resolve({ status: "ok" });
    }
    return Promise.reject(new Error(`unexpected endpoint: ${url}`));
  };
}

async function renderLoaded(overrides: Partial<Record<string, unknown>> = {}) {
  requestMock.mockImplementation(makeRoute(overrides) as never);
  const view = render(<StudentKrm />);
  const show = screen.getByRole("button", { name: "Показать" });
  await waitFor(() => expect(show).toBeEnabled());
  return view;
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

beforeEach(() => {
  requestMock.mockReset();
  // reportError -> POST /api/log: заглушка вместо реальной сети.
  vi.stubGlobal(
    "fetch",
    vi.fn(() => Promise.resolve(new Response("{}", { status: 200 }))),
  );
  vi.spyOn(console, "error").mockImplementation(() => {});
});

describe("StudentKrm. Первичная отрисовка", () => {
  it("рисует заголовок и подсказку до загрузки программы", async () => {
    requestMock.mockImplementation(makeRoute() as never);
    render(<StudentKrm />);

    expect(screen.getByText("Мои компетенции: программа и я")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Сравнение программы с вашими навыками - компактный вид по дисциплинам",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Сначала нажмите «Показать», чтобы загрузить программу"),
    ).toBeInTheDocument();

    // Направления и профили запрашиваются при монтировании.
    await waitFor(() =>
      expect(requestMock).toHaveBeenCalledWith("/api/krm/directions"),
    );
    expect(requestMock).toHaveBeenCalledWith("/api/profiles");
    // Первое направление становится выбранным: кнопка разблокирована.
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Показать" })).toBeEnabled(),
    );
  });

  it("показывает статистику и дисциплины после «Показать»", async () => {
    const user = userEvent.setup();
    await renderLoaded();

    await user.click(screen.getByRole("button", { name: "Показать" }));

    expect(
      await screen.findByText("Информатика и вычислительная техника"),
    ).toBeInTheDocument();
    // Плитки статистики: значения и подписи.
    expect(screen.getByText("КРМ")).toBeInTheDocument();
    expect(screen.getByText("Мои")).toBeInTheDocument();
    expect(screen.getByText("Совпало")).toBeInTheDocument();
    expect(screen.getByText("Только мои")).toBeInTheDocument();
    expect(screen.getByText("Показано 2 из 2 дисциплин")).toBeInTheDocument();

    expect(screen.getByText("Базы данных")).toBeInTheDocument();
    expect(screen.getByText("Сети")).toBeInTheDocument();
    expect(screen.getByText("1/2")).toBeInTheDocument();

    // Раскрываем дисциплину: код компетенции и оба навыка.
    await user.click(screen.getByText("Базы данных"));
    expect(await screen.findByText("ПК-1")).toBeInTheDocument();
    expect(screen.getByText("SQL")).toBeInTheDocument();
    expect(screen.getByText("Индексы")).toBeInTheDocument();
    expect(screen.getByTitle("Есть у вас")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Добавить «Индексы»" }),
    ).toBeInTheDocument();

    // Сворачиваем компетенцию: навыки исчезают.
    await user.click(screen.getByText("ПК-1"));
    await waitFor(() => expect(screen.queryByText("SQL")).toBeNull());
  });
});

describe("StudentKrm. Фильтры", () => {
  it("фильтрует дисциплины по строке и сбрасывает запрос", async () => {
    const user = userEvent.setup();
    await renderLoaded();
    await user.click(screen.getByRole("button", { name: "Показать" }));
    await screen.findByText("Базы данных");

    const filter = screen.getByLabelText("Фильтр дисциплин");
    await user.type(filter, "сети");
    expect(screen.getByText("Показано 1 из 2 дисциплин")).toBeInTheDocument();
    expect(screen.queryByText("Базы данных")).toBeNull();

    await user.click(screen.getByRole("button", { name: "Очистить фильтр" }));
    expect(screen.getByText("Показано 2 из 2 дисциплин")).toBeInTheDocument();

    await user.type(filter, "нет такого");
    expect(
      screen.getByText("По фильтру ничего не найдено - измените запрос или сбросьте фильтр"),
    ).toBeInTheDocument();
  });

  it("переключатель совпадений оставляет только дисциплины с совпадениями", async () => {
    const user = userEvent.setup();
    await renderLoaded();
    await user.click(screen.getByRole("button", { name: "Показать" }));
    await screen.findByText("Сети");

    await user.click(screen.getByRole("combobox", { name: "Фильтр совпадений" }));
    await user.click(await screen.findByRole("option", { name: "Есть совпадения" }));

    expect(screen.getByText("Показано 1 из 2 дисциплин")).toBeInTheDocument();
    expect(screen.getByText("Базы данных")).toBeInTheDocument();
    expect(screen.queryByText("Сети")).toBeNull();
  });

  it("«Развернуть» и «Свернуть» раскрывают все дисциплины сразу", async () => {
    const user = userEvent.setup();
    await renderLoaded();
    await user.click(screen.getByRole("button", { name: "Показать" }));
    await screen.findByText("Сети");

    await user.click(screen.getByRole("button", { name: "Развернуть" }));
    expect(await screen.findByText("ПК-2")).toBeInTheDocument();
    expect(screen.getByText("ПК-1")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Свернуть" }));
    await waitFor(() => expect(screen.queryByText("ПК-1")).toBeNull());
  });
});

describe("StudentKrm. Ветки ошибок и пустых данных", () => {
  it("показывает ошибку загрузки программы", async () => {
    const user = userEvent.setup();
    requestMock.mockImplementation(
      ((url: string) =>
        url.startsWith("/api/krm/student/competencies")
          ? Promise.reject(new Error("Нет доступа к КРМ"))
          : makeRoute()(url)) as never,
    );
    render(<StudentKrm />);
    const show = screen.getByRole("button", { name: "Показать" });
    await waitFor(() => expect(show).toBeEnabled());
    await user.click(show);

    expect(await screen.findByText("Нет доступа к КРМ")).toBeInTheDocument();
  });

  it("пустая программа показывает подсказку «нет данных»", async () => {
    const user = userEvent.setup();
    await renderLoaded({ krm: EMPTY_KRM });
    await user.click(screen.getByRole("button", { name: "Показать" }));

    expect(
      await screen.findByText("нет данных - добавьте навыки ниже"),
    ).toBeInTheDocument();
    expect(screen.getByText("Показано 0 из 0 дисциплин")).toBeInTheDocument();
    expect(screen.getByText("Всё из программы уже у вас")).toBeInTheDocument();
  });
});

describe("StudentKrm. Добавление навыков", () => {
  it("добавляет свой навык кнопкой и по Enter, показывает сообщение", async () => {
    const user = userEvent.setup();
    await renderLoaded();
    await user.click(screen.getByRole("button", { name: "Показать" }));
    await screen.findByText("Базы данных");

    const input = screen.getByLabelText("Новый навык");
    const add = screen.getByRole("button", { name: "Добавить навык" });
    expect(add).toBeDisabled();
    await user.type(input, "Docker");
    expect(add).toBeEnabled();
    await user.click(add);

    await waitFor(() =>
      expect(requestMock).toHaveBeenCalledWith("/api/profiles/self", {
        method: "PATCH",
        body: { add_skills: ["Docker"] },
      }),
    );
    expect(
      await screen.findByText("«Docker» - добавлено в ваши навыки"),
    ).toBeInTheDocument();
    expect(screen.getByText("Добавлено вами в этой сессии (1)")).toBeInTheDocument();
    expect(screen.getByText("Docker")).toBeInTheDocument();

    // Второй навык - клавишей Enter.
    await user.type(input, "Kafka{Enter}");
    await waitFor(() =>
      expect(requestMock).toHaveBeenCalledWith("/api/profiles/self", {
        method: "PATCH",
        body: { add_skills: ["Kafka"] },
      }),
    );
    expect(screen.getByText("Добавлено вами в этой сессии (2)")).toBeInTheDocument();
  });

  it("добавляет компетенцию из программы кнопкой +", async () => {
    const user = userEvent.setup();
    await renderLoaded();
    await user.click(screen.getByRole("button", { name: "Показать" }));
    await user.click(await screen.findByText("Базы данных"));

    await user.click(await screen.findByRole("button", { name: "Добавить «Индексы»" }));

    await waitFor(() =>
      expect(requestMock).toHaveBeenCalledWith("/api/profiles/self", {
        method: "PATCH",
        body: { add_skills: ["Индексы"] },
      }),
    );
    expect(
      await screen.findByText("«Индексы» - добавлено в ваши навыки"),
    ).toBeInTheDocument();
  });

  it("ошибка PATCH возвращает прежние данные и текст ошибки", async () => {
    const user = userEvent.setup();
    await renderLoaded({ patchRejects: true });
    await user.click(screen.getByRole("button", { name: "Показать" }));
    await screen.findByText("Базы данных");

    await user.type(screen.getByLabelText("Новый навык"), "Kafka");
    await user.click(screen.getByRole("button", { name: "Добавить навык" }));

    expect(await screen.findByText("Сервер недоступен")).toBeInTheDocument();
    // Ввод восстановлен, оптимистичная вставка откатана.
    expect(screen.getByLabelText("Новый навык")).toHaveValue("Kafka");
    expect(screen.queryByText("Добавлено вами в этой сессии (1)")).toBeNull();
  });
});

describe("StudentKrm. Навигация", () => {
  it("кнопка «Открыть мой профиль» вызывает onNavigate('profile')", async () => {
    const user = userEvent.setup();
    const onNavigate = vi.fn();
    requestMock.mockImplementation(makeRoute() as never);
    render(<StudentKrm onNavigate={onNavigate} />);

    const btn = screen.getByRole("button", { name: "Открыть мой профиль" });
    await user.click(btn);
    expect(onNavigate).toHaveBeenCalledWith("profile");
    // Кнопка «Показать» рядом не мешает: она в шапке карточки.
    expect(within(document.body).getByRole("button", { name: "Показать" })).toBeInTheDocument();
  });
});
