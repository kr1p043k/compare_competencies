/**
 * Тесты docked-панели чата с ассистентом (LlmChatPanel).
 *
 * Сеть и SSE подменены: request - vi.fn() из мокованного lib/api (весь
 * модуль, чтобы не тянуть auth-заголовки в jsdom), EventSource - фейковый
 * класс с реестром инстансов и emit(). Проверяется контракт панели:
 * история при открытии (и 404 как пустой чат), параметры URL стрима
 * (student: только message; teacher: + discipline), накопление
 * text-дельт, рендер block-метрики, завершение по done, ошибка с
 * «Повторить», отключённый чат у преподавателя без дисциплины.
 */

import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError, request } from "../../lib/api";
import { reportError } from "../../lib/errors";
import { LlmChatPanel, parseHistory } from "./LlmChatPanel";

// instanceof ApiError в панели должен работать на мок-классе: класс
// объявлен внутри фабрики, поэтому панель и тест видят один и тот же.
vi.mock("../../lib/api", () => {
  class ApiError extends Error {
    readonly status: number;
    constructor(message: string, status: number) {
      super(message);
      this.name = "ApiError";
      this.status = status;
    }
  }
  return { ApiError, request: vi.fn() };
});

// reportError шлёт POST /api/log - в jsdom сетевой слой не нужен.
vi.mock("../../lib/errors", () => ({
  reportError: vi.fn(),
  onError: vi.fn(() => () => {}),
}));

type SseListener = (event: Event) => void;

/** Фейковый EventSource: помнит URL, слушателей и флаг close. */
class FakeEventSource {
  static instances: FakeEventSource[] = [];

  readonly url: string;
  closed = false;
  private readonly listeners = new Map<string, Set<SseListener>>();

  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
  }

  addEventListener(type: string, listener: SseListener): void {
    const set = this.listeners.get(type) ?? new Set<SseListener>();
    set.add(listener);
    this.listeners.set(type, set);
  }

  close(): void {
    this.closed = true;
  }

  /** Событие с data (именованный SSE) или без (разрыв соединения). */
  emit(type: string, data?: string): void {
    const set = this.listeners.get(type);
    if (!set) return;
    const event = (data === undefined ? { type } : { type, data }) as unknown as MessageEvent;
    for (const listener of [...set]) listener(event as unknown as Event);
  }
}

/** Последний созданный стрим (к моменту вызова act-фаза уже прошла). */
function lastStream(): FakeEventSource {
  const es = FakeEventSource.instances.at(-1);
  expect(es).toBeDefined();
  return es as FakeEventSource;
}

const FAB = { name: "Открыть чат с ассистентом" };
const SEND = { name: "Отправить сообщение" };

beforeEach(() => {
  FakeEventSource.instances = [];
  vi.stubGlobal("EventSource", FakeEventSource);
  vi.mocked(request).mockReset();
  vi.mocked(request).mockResolvedValue(undefined);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("LlmChatPanel", () => {
  it("открывается FAB-ом и грузит историю сообщений (student)", async () => {
    const user = userEvent.setup();
    // Форма истории как в llm_chat: messages + blocks_json-словарь ChatMessage.
    vi.mocked(request).mockResolvedValueOnce({
      messages: [
        { role: "user", content: "Что с покрытием?" },
        {
          role: "assistant",
          blocks_json: {
            blocks: [
              { type: "text", content: "Сводка по рынку" },
              { type: "metric", name: "pokrytie", value: 82 },
            ],
          },
        },
      ],
    });

    render(<LlmChatPanel role="student" />);
    await user.click(screen.getByRole("button", FAB));

    // Панель-aside появилась, FAB скрылся.
    expect(screen.getByRole("complementary", { name: "Чат с ассистентом" })).toBeInTheDocument();
    expect(screen.queryByRole("button", FAB)).not.toBeInTheDocument();

    // Обе реплики истории отрисованы, метрика - StatCard-плиткой.
    await screen.findByText("Что с покрытием?");
    expect(screen.getByText("Сводка по рынку")).toBeInTheDocument();
    expect(screen.getByText("pokrytie")).toBeInTheDocument();
    expect(screen.getByText("82")).toBeInTheDocument();

    // У студента история без параметров.
    expect(vi.mocked(request).mock.calls[0][0]).toBe("/api/student/chat/history");
  });

  it("teacher: история запрашивается с параметром дисциплины", async () => {
    const user = userEvent.setup();
    vi.mocked(request).mockResolvedValueOnce({ messages: [] });

    render(
      <LlmChatPanel role="teacher" discipline="Базы данных" contextLabel="Базы данных" />,
    );
    await user.click(screen.getByRole("button", FAB));

    await waitFor(() => expect(request).toHaveBeenCalledTimes(1));
    const url = new URL(vi.mocked(request).mock.calls[0][0], "http://localhost");
    expect(url.pathname).toBe("/api/teacher/chat/history");
    expect(url.searchParams.get("discipline")).toBe("Базы данных");

    // Подпись контекста видна в шапке панели.
    expect(screen.getByTitle("Базы данных")).toBeInTheDocument();
  });

  it("404 истории - пустой чат, а не ошибка", async () => {
    const user = userEvent.setup();
    vi.mocked(request).mockRejectedValueOnce(new ApiError("Данные не найдены", 404, null, "/api/student/chat/history"));

    render(<LlmChatPanel role="student" />);
    await user.click(screen.getByRole("button", FAB));

    await screen.findByText("Спросите ассистента");
    expect(screen.queryByText("Не удалось загрузить историю переписки")).not.toBeInTheDocument();
    expect(reportError).not.toHaveBeenCalled();
  });

  it("сбой истории - ErrorState, повтор перезапрашивает", async () => {
    const user = userEvent.setup();
    vi.mocked(request).mockRejectedValueOnce(new ApiError("Нет связи с сервером", 0, null, "/api/student/chat/history"));

    render(<LlmChatPanel role="student" />);
    await user.click(screen.getByRole("button", FAB));

    await screen.findByText("Не удалось загрузить историю переписки");
    expect(reportError).toHaveBeenCalledWith("LlmChatPanel:history", expect.anything());

    // Повтор в ErrorState ведёт к повторному запросу истории.
    await user.click(screen.getByRole("button", { name: "Повторить" }));
    await screen.findByText("Спросите ассистента");
    expect(request).toHaveBeenCalledTimes(2);
  });

  it("teacher без дисциплины: отправка выключена, история не грузится", async () => {
    const user = userEvent.setup();
    render(<LlmChatPanel role="teacher" discipline={null} />);
    await user.click(screen.getByRole("button", FAB));

    const send = await screen.findByRole("button", SEND);
    expect(send).toBeDisabled();

    const input = screen.getByLabelText("Сообщение ассистенту");
    expect(input).toHaveAttribute("placeholder", "Сначала выберите дисциплину");
    expect(
      screen.getByText("Выберите дисциплину в списке слева, чтобы задать вопрос ассистенту."),
    ).toBeInTheDocument();

    await user.type(input, "Вопрос без дисциплины");
    expect(send).toBeDisabled();
    expect(request).not.toHaveBeenCalled();
    expect(FakeEventSource.instances).toHaveLength(0);
  });

  it("student стрим: Enter отправляет, дельты копятся, metric-блок, done закрывает", async () => {
    const user = userEvent.setup();
    render(<LlmChatPanel role="student" />);
    await user.click(screen.getByRole("button", FAB));

    const input = await screen.findByLabelText("Сообщение ассистенту");
    await user.type(input, "Как дела?");
    await user.type(input, "{enter}");

    const es = lastStream();
    const url = new URL(es.url, "http://localhost");
    expect(url.pathname).toBe("/api/student/chat/stream");
    expect(url.searchParams.get("message")).toBe("Как дела?");
    expect(url.searchParams.get("discipline")).toBeNull();
    // Поле ввода очистилось, вопрос ушёл в ленту.
    expect(screen.getByLabelText("Сообщение ассистенту")).toHaveValue("");
    expect(screen.getByText("Как дела?")).toBeInTheDocument();

    // Дельты: первая приходит JSON-литералом строки, вторая - сырой.
    act(() => es.emit("text", JSON.stringify("Привет")));
    act(() => es.emit("text", " мир"));
    expect(screen.getByText("Привет мир")).toBeInTheDocument();
    expect(screen.getByText("Ассистент отвечает")).toBeInTheDocument();

    // Структурный block: метрика рисуется StatCard-плиткой.
    act(() => es.emit("block", JSON.stringify({ type: "metric", name: "coverage", value: 82 })));
    expect(screen.getByText("coverage")).toBeInTheDocument();
    expect(screen.getByText("82")).toBeInTheDocument();

    act(() => es.emit("done", JSON.stringify({ thread_id: "t-1", message_id: "m-1" })));
    expect(es.closed).toBe(true);
    expect(screen.queryByText("Ассистент отвечает")).not.toBeInTheDocument();
  });

  it("teacher стрим: URL содержит дисциплину", async () => {
    const user = userEvent.setup();
    render(<LlmChatPanel role="teacher" discipline="Базы данных" />);
    await user.click(screen.getByRole("button", FAB));

    const input = await screen.findByLabelText("Сообщение ассистенту");
    await user.type(input, "Что с пробелами?");
    await user.click(screen.getByRole("button", SEND));

    const url = new URL(lastStream().url, "http://localhost");
    expect(url.pathname).toBe("/api/teacher/chat/stream");
    expect(url.searchParams.get("discipline")).toBe("Базы данных");
    expect(url.searchParams.get("message")).toBe("Что с пробелами?");
  });

  it("error-событие: алерт + Повторить переспрашивает тот же вопрос", async () => {
    const user = userEvent.setup();
    render(<LlmChatPanel role="student" />);
    await user.click(screen.getByRole("button", FAB));

    const input = await screen.findByLabelText("Сообщение ассистенту");
    await user.type(input, "Что покрыто?");
    await user.click(screen.getByRole("button", SEND));
    const es1 = lastStream();

    act(() => es1.emit("error", JSON.stringify({ message: "LLM перегружен" })));
    expect(es1.closed).toBe(true);

    const alert = screen.getByRole("alert");
    expect(within(alert).getByText("LLM перегружен")).toBeInTheDocument();

    await user.click(within(alert).getByRole("button", { name: "Повторить" }));
    const es2 = lastStream();
    expect(es2).not.toBe(es1);
    // Повтор уходит с тем же вопросом, упавшая реплика удалена.
    const url = new URL(es2.url, "http://localhost");
    expect(url.searchParams.get("message")).toBe("Что покрыто?");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();

    act(() => es2.emit("text", "Второй ответ"));
    act(() => es2.emit("done", JSON.stringify({ thread_id: "t", message_id: "m" })));
    expect(screen.getByText("Второй ответ")).toBeInTheDocument();
    expect(screen.getAllByText("Что покрыто?")).toHaveLength(1);
  });

  it("обрыв без единого события - понятная ошибка", async () => {
    const user = userEvent.setup();
    render(<LlmChatPanel role="student" />);
    await user.click(screen.getByRole("button", FAB));

    const input = await screen.findByLabelText("Сообщение ассистенту");
    await user.type(input, "Вечный вопрос");
    await user.click(screen.getByRole("button", SEND));

    const es = lastStream();
    act(() => es.emit("error")); // разрыв соединения: data нет
    expect(screen.getByText("Соединение прервано. Попробуйте снова.")).toBeInTheDocument();
    expect(es.closed).toBe(true);
  });

  it("обрыв после частичного ответа - тихое завершение без алерта", async () => {
    const user = userEvent.setup();
    render(<LlmChatPanel role="student" />);
    await user.click(screen.getByRole("button", FAB));

    const input = await screen.findByLabelText("Сообщение ассистенту");
    await user.type(input, "Ещё вопрос");
    await user.click(screen.getByRole("button", SEND));

    const es = lastStream();
    act(() => es.emit("text", "Частичный ответ"));
    act(() => es.emit("error"));
    expect(es.closed).toBe(true);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByText("Частичный ответ")).toBeInTheDocument();
  });
});

describe("parseHistory", () => {
  it("разбирает messages + blocks_json-словарь: текст склеен, метрика блоком", () => {
    const result = parseHistory({
      messages: [
        { role: "user", content: "Что с покрытием?" },
        {
          role: "assistant",
          blocks_json: {
            blocks: [
              { type: "text", content: "Сводка" },
              { type: "metric", name: "pokrytie", value: 82 },
            ],
          },
        },
      ],
    });

    expect(result).toHaveLength(2);
    expect(result[0]).toMatchObject({ role: "user", text: "Что с покрытием?" });
    expect(result[1].role).toBe("assistant");
    expect(result[1].text).toBe("Сводка");
    expect(result[1].blocks).toEqual([{ type: "metric", name: "pokrytie", value: 82 }]);
  });

  it("голый массив, {history} и {thread.messages} читаются одинаково", () => {
    expect(parseHistory([{ role: "assistant", text: "раз" }])).toEqual([
      expect.objectContaining({ role: "assistant", text: "раз" }),
    ]);
    expect(parseHistory({ history: [{ role: "user", text: "два" }] })[0].text).toBe("два");
    expect(parseHistory({ thread: { messages: [{ role: "assistant", content: "три" }] } })[0].text).toBe("три");
  });

  it("blocks_json {text: ...} + точные дубли текстовых блоков не дублируются", () => {
    const result = parseHistory([
      {
        role: "assistant",
        blocks_json: { text: "Ответ" },
        blocks: [
          { type: "text", content: "Ответ" }, // дубль основного текста
          { type: "text", content: "Позже" }, // новое - приклеивается
        ],
      },
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].text).toBe("Ответ\n\nПозже");
  });

  it("мусор -> пустая история", () => {
    expect(parseHistory(null)).toEqual([]);
    expect(parseHistory({})).toEqual([]);
    expect(parseHistory({ messages: "не-массив" })).toEqual([]);
    expect(parseHistory([{ garbage: true }, 42, "строка"])).toEqual([]);
  });
});
