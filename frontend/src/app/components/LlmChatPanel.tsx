/**
 * Docked-панель чата с LLM-ассистентом (роль teacher или student).
 *
 * Бэкендовый контракт (роутеры /api/{teacher|student}/chat/*):
 *  - стрим:   GET SSE  /api/teacher/chat/stream?discipline=<code>&message=<text>
 *                       /api/student/chat/stream?message=<text>
 *  - история: GET       /api/teacher/chat/history?discipline=<code>
 *                       /api/student/chat/history
 *  - SSE-события: "text" (строка-дельта), "block" (JSON блока:
 *    type=text|metric|citation|chart|action_list), "done"
 *    ({"thread_id","message_id"}), "error" ({"message"}).
 *
 * История пишется параллельно с этим фронтом, поэтому её разбор
 * устойчив к нескольким формам обёртки ({messages}|{history}|[...]) и к
 * блокам ChatMessage.blocks_json (роль - "user"|"assistant"). Чего в
 * payload нет - того не придумываем: пустые поля не рендерятся.
 */

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type FormEvent,
  type ReactNode,
} from "react";
import { RotateCcw, Send, Sparkles, X } from "lucide-react";
import { Badge, Button, EmptyState, ErrorState, Skeleton } from "../../design";
import { ApiError, request } from "../../lib/api";
import { reportError } from "../../lib/errors";
import {
  ChatBlockView,
  extractBlockText,
  normalizeBlock,
  type ChatBlock,
} from "./ChatBlockView";
import { ChatMarkdown } from "./ChatMarkdown";
import { cn } from "./ui/utils";

export type ChatRole = "teacher" | "student";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  /** Накопленный текст: SSE-дельты "text" + текстовые блоки. */
  text: string;
  /** Структурные блоки (кроме text) в порядке прихода. */
  blocks: ChatBlock[];
  /** Отметка «ответ ещё идёт» для индикатора. */
  streaming?: boolean;
  /** Ошибка ответа: показывается с кнопкой «Повторить». */
  error?: string | null;
}

export interface LlmChatPanelProps {
  role: ChatRole;
  /**
   * Контекст преподавателя: код направления (dir_code, например «09.03.02») -
   * бэкенд llm_chat читает teacher/<код>/_summary.json и
   * krm_disciplines_<код>.json; человеческое имя даёт пустой контекст.
   * Пусто - нет чата: кнопка отправки недоступна, история не грузится.
   */
  discipline?: string | null;
  /** Человекочитаемая подпись контекста в шапке панели. */
  contextLabel?: string | null;
}

const STREAM_URL: Record<ChatRole, string> = {
  teacher: "/api/teacher/chat/stream",
  student: "/api/student/chat/stream",
};

const HISTORY_URL: Record<ChatRole, string> = {
  teacher: "/api/teacher/chat/history",
  student: "/api/student/chat/history",
};

const PANEL_ID = "llm-chat-panel";

/* --------------------------------------------------------------------- утилиты */

let msgSeq = 0;
function nextMsgId(): string {
  msgSeq += 1;
  return `llm-msg-${msgSeq}`;
}

function strOf(value: unknown): string {
  return typeof value === "string" ? value : "";
}

/** SSE-данные события: MessageEvent.data при штатной работе. */
function dataOf(event: Event | MessageEvent): string | null {
  const data = (event as MessageEvent).data;
  return typeof data === "string" ? data : null;
}

/** Дельта "text" может прийти как сырая строка или JSON-литерал строки. */
function decodeTextDelta(data: string): string {
  const trimmed = data.trim();
  if (trimmed.startsWith('"')) {
    try {
      const parsed: unknown = JSON.parse(trimmed);
      if (typeof parsed === "string") return parsed;
    } catch {
      /* сырая строка с кавычкой в начале - показываем как есть */
    }
  }
  return data;
}

function parseEventJson<T>(data: string | null): T | null {
  if (!data) return null;
  try {
    return JSON.parse(data) as T;
  } catch {
    return null;
  }
}

function extractHistoryItems(raw: unknown): unknown[] | null {
  if (Array.isArray(raw)) return raw;
  if (raw && typeof raw === "object") {
    const rec = raw as Record<string, unknown>;
    for (const key of ["messages", "history", "items"]) {
      if (Array.isArray(rec[key])) return rec[key] as unknown[];
    }
    const thread = rec.thread;
    if (thread && typeof thread === "object" && !Array.isArray(thread)) {
      const nested = (thread as Record<string, unknown>).messages;
      if (Array.isArray(nested)) return nested as unknown[];
    }
  }
  return null;
}

/**
 * История -> представление панели. Форма придумана не нами, поэтому:
 * обёртка {messages|history|items|thread.messages} или голый массив;
 * каждый элемент - {role, text?, blocks_json?, blocks?, content?};
 * blocks_json - словарь ({"blocks": [...]} либо {"text": ...}) как в
 * модели ChatMessage. Текстовые блоки аккуратно склеиваются в тело
 * реплики, структурные остаются блоками.
 */
export function parseHistory(raw: unknown): ChatMessage[] {
  const items = extractHistoryItems(raw);
  if (!items) return [];

  const out: ChatMessage[] = [];
  for (const item of items) {
    if (!item || typeof item !== "object" || Array.isArray(item)) continue;
    const rec = item as Record<string, unknown>;
    const role: "user" | "assistant" = rec.role === "user" ? "user" : "assistant";

    const bjson = rec.blocks_json ?? rec.blocksJson;
    const rawBlocks: unknown[] = Array.isArray(bjson)
      ? (bjson as unknown[])
      : bjson && typeof bjson === "object" && Array.isArray((bjson as Record<string, unknown>).blocks)
        ? ((bjson as Record<string, unknown>).blocks as unknown[])
        : Array.isArray(rec.blocks)
          ? (rec.blocks as unknown[])
          : [];

    let text =
      strOf(rec.text) ||
      (bjson && typeof bjson === "object" && !Array.isArray(bjson)
        ? strOf((bjson as Record<string, unknown>).text)
        : "") ||
      strOf(rec.content);

    const blocks: ChatBlock[] = [];
    for (const rawBlock of rawBlocks) {
      const block = normalizeBlock(rawBlock);
      if (!block) continue;
      if (block.type === "text") {
        const t = extractBlockText(block);
        if (t && !text.includes(t)) text = text ? `${text}\n\n${t}` : t;
        continue;
      }
      blocks.push(block);
    }

    if (!text && blocks.length === 0) continue;
    out.push({ id: nextMsgId(), role, text, blocks });
  }
  return out;
}

/* --------------------------------------------------------------- реплика чата */

function MessageView({
  message,
  onRetry,
}: {
  message: ChatMessage;
  onRetry: (id: string) => void;
}) {
  const isUser = message.role === "user";
  return (
    <div className={cn("flex", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[92%] space-y-2 rounded-2xl px-3.5 py-2.5 text-sm",
          isUser
            ? "bg-primary text-primary-foreground"
            : "border border-border bg-card text-foreground",
        )}
      >
        {message.text ? <ChatMarkdown>{message.text}</ChatMarkdown> : null}
        {message.blocks.map((block, i) => (
          <ChatBlockView key={`${message.id}-b${i}`} block={block} />
        ))}
        {message.streaming && (
          <span className="flex items-center gap-1 pt-0.5" aria-hidden="true">
            {["[animation-delay:0s]", "[animation-delay:150ms]", "[animation-delay:300ms]"].map((delayClass) => (
              <span
                key={delayClass}
                className={cn(
                  "size-1.5 animate-pulse rounded-full bg-muted-foreground/70 motion-reduce:animate-none",
                  delayClass
                )}
              />
            ))}
          </span>
        )}
        {message.error && (
          <div
            role="alert"
            className="rounded-lg border border-destructive/30 bg-destructive/5 px-2.5 py-2 text-xs text-destructive"
          >
            <div className="whitespace-pre-wrap">{message.error}</div>
            <Button
              variant="secondary"
              size="sm"
              className="mt-2"
              iconLeft={<RotateCcw className="size-3.5" aria-hidden="true" />}
              onClick={() => onRetry(message.id)}
            >
              Повторить
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}

/* --------------------------------------------------------------------- панель */

export function LlmChatPanel({ role, discipline, contextLabel }: LlmChatPanelProps) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [historyError, setHistoryError] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);

  const esRef = useRef<EventSource | null>(null);
  const finishedRef = useRef(true);
  const hasContentRef = useRef(false);
  const threadIdRef = useRef<string | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  const disciplineParam = discipline ?? "";
  const canChat = role === "student" || disciplineParam !== "";

  /* ------------------------------------------------------------- стрим-логика */

  const closeStream = useCallback(() => {
    finishedRef.current = true;
    setStreaming(false);
    esRef.current?.close();
    esRef.current = null;
  }, []);

  const setAssistant = useCallback((assistantId: string, patch: (m: ChatMessage) => ChatMessage) => {
    setMessages((prev) => prev.map((m) => (m.id === assistantId ? patch(m) : m)));
  }, []);

  const finishAssistant = useCallback(
    (assistantId: string) => {
      closeStream();
      setAssistant(assistantId, (m) => ({ ...m, streaming: false }));
    },
    [closeStream, setAssistant],
  );

  const failAssistant = useCallback(
    (assistantId: string, message: string) => {
      closeStream();
      setAssistant(assistantId, (m) => ({ ...m, streaming: false, error: message }));
    },
    [closeStream, setAssistant],
  );

  const startStream = useCallback(
    (question: string, assistantId: string) => {
      const params = new URLSearchParams();
      if (role === "teacher") params.set("discipline", disciplineParam);
      params.set("message", question);

      finishedRef.current = false;
      hasContentRef.current = false;
      setStreaming(true);

      let es: EventSource;
      try {
        es = new EventSource(`${STREAM_URL[role]}?${params.toString()}`);
      } catch {
        failAssistant(assistantId, "Не удалось открыть поток ответа");
        return;
      }
      esRef.current = es;

      es.addEventListener("text", (event) => {
        const data = dataOf(event);
        if (data === null) return;
        const delta = decodeTextDelta(data);
        if (!delta) return;
        hasContentRef.current = true;
        setAssistant(assistantId, (m) => ({ ...m, text: m.text + delta }));
      });

      es.addEventListener("block", (event) => {
        const block = normalizeBlock(parseEventJson<unknown>(dataOf(event)));
        if (!block) return;
        hasContentRef.current = true;
        if (block.type === "text") {
          const text = extractBlockText(block);
          // точный дубль уже стримившегося текста не дублируем
          setAssistant(assistantId, (m) =>
            text && !m.text.includes(text)
              ? { ...m, text: m.text + (m.text && !m.text.endsWith("\n") ? "\n\n" : "") + text }
              : m,
          );
          return;
        }
        setAssistant(assistantId, (m) => ({ ...m, blocks: [...m.blocks, block] }));
      });

      es.addEventListener("done", (event) => {
        const payload = parseEventJson<{ thread_id?: unknown }>(dataOf(event));
        if (payload && typeof payload.thread_id === "string") {
          threadIdRef.current = payload.thread_id;
        }
        finishAssistant(assistantId);
      });

      // Именованное событие "error" ({"message"}) и разрыв соединения
      // приходят в один listener: у разрыва нет data.
      es.addEventListener("error", (event) => {
        if (finishedRef.current) return;
        const payload = parseEventJson<{ message?: unknown }>(dataOf(event));
        if (payload && typeof payload.message === "string") {
          failAssistant(assistantId, payload.message);
          return;
        }
        if (hasContentRef.current) {
          // сервер закончил поток без "done", но ответ уже пришёл - не пугаем
          finishAssistant(assistantId);
        } else {
          failAssistant(assistantId, "Соединение прервано. Попробуйте снова.");
        }
      });
    },
    [disciplineParam, failAssistant, finishAssistant, role, setAssistant],
  );

  const handleSend = useCallback(
    (questionRaw: string) => {
      const question = questionRaw.trim();
      if (!question || streaming || !canChat) return;
      closeStream(); // страховка: второго параллельного потока быть не должно
      const assistantId = nextMsgId();
      hasContentRef.current = false;
      setMessages((prev) => [
        ...prev,
        { id: nextMsgId(), role: "user", text: question, blocks: [] },
        { id: assistantId, role: "assistant", text: "", blocks: [], streaming: true },
      ]);
      startStream(question, assistantId);
    },
    [canChat, closeStream, startStream, streaming],
  );

  const handleRetry = useCallback(
    (failedId: string) => {
      if (streaming || !canChat) return;
      // вопрос - последняя пользовательская реплика перед упавшим ответом
      let question = "";
      const idx = messages.findIndex((m) => m.id === failedId);
      for (let i = (idx >= 0 ? idx : messages.length) - 1; i >= 0; i -= 1) {
        if (messages[i].role === "user") {
          question = messages[i].text;
          break;
        }
      }
      if (!question) return;
      const assistantId = nextMsgId();
      setMessages((prev) => [
        ...prev.filter((m) => m.id !== failedId),
        { id: assistantId, role: "assistant", text: "", blocks: [], streaming: true },
      ]);
      startStream(question, assistantId);
    },
    [canChat, messages, startStream, streaming],
  );

  /* ----------------------------------------------------------------- история */

  const loadHistory = useCallback(async () => {
    if (role === "teacher" && !disciplineParam) return;
    // смена контекста во время стрима: старый поток больше не актуален
    closeStream();
    setMessages([]);
    setHistoryError(false);
    setHistoryLoading(true);
    try {
      const url =
        role === "teacher"
          ? `${HISTORY_URL[role]}?${new URLSearchParams({ discipline: disciplineParam })}`
          : HISTORY_URL[role];
      const raw = await request<unknown>(url);
      setMessages(parseHistory(raw));
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        // треда ещё нет - это не ошибка, просто пустой чат
      } else {
        reportError("LlmChatPanel:history", err);
        setHistoryError(true);
      }
    } finally {
      setHistoryLoading(false);
    }
  }, [closeStream, disciplineParam, role]);

  // История грузится при каждом открытии панели и при смене дисциплины.
  useEffect(() => {
    if (!open || !canChat) return;
    void loadHistory();
  }, [canChat, loadHistory, open]);

  // Размонтирование панели (смена вкладки) не должно оставлять открытый SSE.
  useEffect(() => () => closeStream(), [closeStream]);

  // Фокус в поле ввода при открытии - можно сразу печатать.
  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  // Автоскролл ленты, но только когда читатель у нижнего края: не дёргаем
  // просмотр старых сообщений.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
    if (nearBottom) el.scrollTop = el.scrollHeight;
  }, [messages, open]);

  /* ------------------------------------------------------------------ ввод */

  const submit = (event: FormEvent) => {
    event.preventDefault();
    const question = input;
    setInput("");
    handleSend(question);
  };

  const onKeyDown = (event: ReactKeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (input.trim() && !streaming && canChat) {
        const question = input;
        setInput("");
        handleSend(question);
      }
    }
  };

  const onClose = () => setOpen(false);

  /* ------------------------------------------------------------------ рендер */

  const canSend = canChat && !streaming && input.trim() !== "";

  let body: ReactNode;
  if (historyLoading && messages.length === 0) {
    body = (
      <div className="space-y-3" aria-hidden="true">
        <Skeleton className="h-12 w-2/3" />
        <Skeleton className="h-20 w-5/6" />
        <Skeleton className="h-14 w-3/4" />
      </div>
    );
  } else if (historyError && messages.length === 0) {
    body = (
      <ErrorState
        title="Не удалось загрузить историю переписки"
        description="Проверьте соединение и попробуйте ещё раз."
        onRetry={() => void loadHistory()}
      />
    );
  } else if (messages.length === 0) {
    body = (
      <EmptyState
        icon={<Sparkles className="size-6" aria-hidden="true" />}
        title="Спросите ассистента"
        description="Ответ приходит потоком: текст, метрики, источники и диаграммы появятся прямо здесь."
      />
    );
  } else {
    body = messages.map((message) => (
      <MessageView key={message.id} message={message} onRetry={handleRetry} />
    ));
  }

  return (
    <>
      {!open && (
        <button
          type="button"
          onClick={() => setOpen(true)}
          aria-label="Открыть чат с ассистентом"
          aria-expanded={false}
          aria-controls={PANEL_ID}
          title="Чат с ассистентом"
          className="fixed bottom-6 right-6 z-[70] inline-flex size-12 items-center justify-center rounded-full bg-primary text-primary-foreground shadow-lg transition-colors hover:brightness-110 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 motion-reduce:transition-none"
        >
          <Sparkles className="size-5" aria-hidden="true" />
        </button>
      )}

      {open && (
        <>
          {/* затемнение закрывает панель тапом - только на узких экранах,
              где панель занимает всю ширину */}
          <div
            className="fixed inset-0 z-[75] bg-black/30 sm:hidden"
            onClick={onClose}
            aria-hidden="true"
          />
          <aside
            id={PANEL_ID}
            aria-label="Чат с ассистентом"
            className="fixed inset-y-0 right-0 z-[80] flex h-full w-full flex-col border-l border-border bg-background shadow-2xl sm:w-[440px]"
            onKeyDown={(event) => {
              if (event.key === "Escape") onClose();
            }}
          >
            <div className="flex items-center gap-2 border-b border-border px-4 py-3">
              <Sparkles className="size-4 shrink-0 text-primary" aria-hidden="true" />
              <h2 className="shrink-0 text-sm font-semibold text-foreground">Ассистент</h2>
              {contextLabel && (
                <Badge tone="neutral" className="min-w-0 max-w-[40%]" title={contextLabel}>
                  <span className="truncate">{contextLabel}</span>
                </Badge>
              )}
              <span role="status" className="sr-only">
                {streaming ? "Ассистент отвечает" : ""}
              </span>
              <Button
                variant="ghost"
                size="sm"
                className="ml-auto h-8 w-8 shrink-0 px-0"
                aria-label="Закрыть чат"
                onClick={onClose}
              >
                <X className="size-4" aria-hidden="true" />
              </Button>
            </div>

            <div
              ref={scrollRef}
              role="log"
              aria-label="Переписка с ассистентом"
              className="min-h-0 flex-1 space-y-3 overflow-y-auto px-4 py-4"
            >
              {body}
            </div>

            <form className="border-t border-border px-4 py-3" onSubmit={submit}>
              {!canChat && (
                <p id="llm-chat-hint" className="mb-2 text-xs text-muted-foreground">
                  Выберите дисциплину в списке слева, чтобы задать вопрос ассистенту.
                </p>
              )}
              <div className="flex items-end gap-2">
                <label htmlFor="llm-chat-input" className="sr-only">
                  Сообщение ассистенту
                </label>
                <textarea
                  id="llm-chat-input"
                  ref={inputRef}
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  onKeyDown={onKeyDown}
                  placeholder={canChat ? "Задайте вопрос ассистенту" : "Сначала выберите дисциплину"}
                  aria-describedby={canChat ? undefined : "llm-chat-hint"}
                  rows={2}
                  className="min-h-10 max-h-32 w-full resize-none rounded-lg border border-input bg-input-background px-3 py-2 text-sm text-foreground placeholder:text-subtle-foreground focus-visible:border-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/30"
                />
                <Button
                  type="submit"
                  size="sm"
                  className="h-10 shrink-0"
                  aria-label="Отправить сообщение"
                  disabled={!canSend}
                >
                  <Send className="size-4" aria-hidden="true" />
                </Button>
              </div>
            </form>
          </aside>
        </>
      )}
    </>
  );
}
