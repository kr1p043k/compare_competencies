/**
 * Единая точка обращения к API.
 *
 * Раньше по проекту было разбросано 61 вызов `fetch` с ручной проверкой
 * `res.ok`, ручным парсингом JSON и собственным форматом ошибок - в каждом
 * файле свой. Здесь: типизированный `request`, класс `ApiError` с
 * человекочитаемым `message` и набор эндпоинтов по доменам.
 */
import { authHeaders, handleUnauthorized } from "./auth";

export class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;
  readonly url: string;

  constructor(message: string, status: number, detail: unknown, url: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.url = url;
  }

  /** 401/403 - сессия истекла или нет прав. */
  get isAuthError(): boolean {
    return this.status === 401 || this.status === 403;
  }

  /** 429 - сработал rate-limit, имеет смысл повторить позже. */
  get isRateLimited(): boolean {
    return this.status === 429;
  }
}

interface RequestOptions extends Omit<RequestInit, "body"> {
  body?: unknown;
  /** Не подставлять Authorization (для /api/auth/*). */
  anonymous?: boolean;
  /** Не парсить ответ как JSON (для файлов). */
  raw?: true;
}

function extractDetail(payload: unknown): string | null {
  if (typeof payload === "string") return payload || null;
  if (!payload || typeof payload !== "object") return null;
  const obj = payload as Record<string, unknown>;
  const detail = obj.detail ?? obj.message ?? obj.error;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const parts = detail
      .map((d) => (typeof d === "object" && d && "msg" in d ? String((d as { msg: unknown }).msg) : null))
      .filter(Boolean);
    return parts.length ? parts.join("; ") : null;
  }
  if (detail && typeof detail === "object") {
    const msg = (detail as Record<string, unknown>).message;
    if (typeof msg === "string") return msg;
  }
  return null;
}

const STATUS_TEXT: Record<number, string> = {
  400: "Некорректный запрос",
  401: "Требуется авторизация",
  403: "Недостаточно прав",
  404: "Данные не найдены",
  409: "Действие уже выполняется",
  422: "Некорректные данные запроса",
  429: "Слишком много запросов, попробуйте позже",
  500: "Внутренняя ошибка сервера",
  503: "Сервис временно недоступен",
};

/**
 * Выполняет запрос и возвращает разобранный JSON.
 * Бросает `ApiError` при любом не-2xx ответе - вызывающий код не должен
 * проверять `res.ok` вручную.
 */
export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { body, anonymous, raw, headers, ...rest } = options;

  const finalHeaders: Record<string, string> = {
    ...(anonymous ? {} : authHeaders()),
    ...((headers as Record<string, string>) ?? {}),
  };

  let payload: BodyInit | undefined;
  if (body instanceof FormData || body instanceof Blob || typeof body === "string") {
    payload = body;
  } else if (body !== undefined) {
    finalHeaders["Content-Type"] = "application/json";
    payload = JSON.stringify(body);
  }

  let res: Response;
  try {
    res = await fetch(path, { ...rest, headers: finalHeaders, body: payload });
  } catch (cause) {
    // Отмена запроса - не сетевой сбой. Пробрасываем как есть: иначе отмена
    // (напр. при быстром переключении вкладок) превращалась бы в «Нет связи
    // с сервером» и вызывающий код не смог бы отличить abort от обрыва.
    if ((cause as { name?: string } | undefined)?.name === "AbortError") throw cause;
    throw new ApiError("Нет связи с сервером", 0, cause, path);
  }

  if (res.status === 401 && !anonymous && !path.includes("/api/auth/login")) {
    handleUnauthorized();
  }

  if (!res.ok) {
    let detail: unknown;
    try {
      detail = await res.json();
    } catch {
      detail = null;
    }
    const message =
      extractDetail(detail) ?? STATUS_TEXT[res.status] ?? `Ошибка запроса (HTTP ${res.status})`;
    throw new ApiError(message, res.status, detail, path);
  }

  if (raw) return res as unknown as T;

  if (res.status === 204) return undefined as T;

  const text = await res.text();
  if (!text) return undefined as T;

  try {
    return JSON.parse(text) as T;
  } catch {
    throw new ApiError("Некорректный ответ сервера", res.status, text, path);
  }
}

/** Скачивание файла: возвращает Blob и имя из Content-Disposition. */
export async function download(
  path: string,
  options: RequestOptions = {},
): Promise<{ blob: Blob; filename: string | null }> {
  const res = await request<Response>(path, { ...options, raw: true });
  const disposition = res.headers.get("content-disposition") ?? "";
  const match = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(disposition);
  return { blob: await res.blob(), filename: match ? decodeURIComponent(match[1]) : null };
}

/** Сохраняет Blob как файл в браузере. */
export function saveBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  URL.revokeObjectURL(url);
  document.body.removeChild(a);
}
