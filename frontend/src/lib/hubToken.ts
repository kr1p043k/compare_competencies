/**
 * Разбор hub-JWT из адреса страницы.
 *
 * hub.sfedu.ru открывает наш сервис с токеном, но формат ссылки не закреплён
 * документацией: встречались и `?token=...`, и `#token=...`. Раньше читался
 * только query, поэтому вход через фрагмент молча не срабатывал: сайт
 * открывался, но сессия не привязывалась к хабу, и вкладки со статьями
 * просили повторный вход.
 *
 * Функции чистые (принимают строки, а не `window.location`), чтобы их можно
 * было покрыть тестами без jsdom.
 */

export const HUB_LOGIN_URL = "https://hub.sfedu.ru/accounts/login/";

const TOKEN_KEY = "token";

export type HubTokenSource = {
  token: string | null;
  /** query без токена: с ведущим «?» либо пустая строка. */
  search: string;
  /** фрагмент без токена: с ведущим «#» либо пустая строка. */
  hash: string;
};

function stripToken(params: URLSearchParams): string {
  params.delete(TOKEN_KEY);
  return params.toString();
}

function cleanSearch(search: string): { value: string; token: string | null } {
  const params = new URLSearchParams(search);
  const token = params.get(TOKEN_KEY);
  if (!token) return { value: search, token: null };
  const rest = stripToken(params);
  return { value: rest ? `?${rest}` : "", token };
}

function cleanHash(hash: string): { value: string; token: string | null } {
  // Фрагмент может быть как чистым `#token=...`, так и маршрутом с query:
  // `#/teacher?token=...`. Маршрут при этом нужно сохранить.
  const body = hash.replace(/^#/, "");
  const marker = body.indexOf("?");
  const route = marker >= 0 ? body.slice(0, marker) : "";
  const raw = marker >= 0 ? body.slice(marker + 1) : body;
  const params = new URLSearchParams(raw);
  const token = params.get(TOKEN_KEY);
  if (!token) return { value: hash, token: null };
  const rest = stripToken(params);
  const value = !route && !rest ? "" : `#${route}${rest ? `?${rest}` : ""}`;
  return { value, token };
}

export function extractHubToken(search: string, hash: string): HubTokenSource {
  const query = cleanSearch(search);
  const fragment = cleanHash(hash);
  const token = query.token ?? fragment.token;
  if (!token) return { token: null, search, hash };
  return { token, search: query.value, hash: fragment.value };
}
