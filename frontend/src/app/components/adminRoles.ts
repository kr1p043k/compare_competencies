/**
 * Общие хелперы админ-консоли: справочник ролей и форматирование дат.
 *
 * Без React-зависимостей. Вкладки «Пользователи» и «Аудит» импортируют
 * отсюда одни и те же подписи и тона ролей, чтобы они не расходились
 * между экранами. fmtDateTime понимает unix-секунды, unix-миллисекунды
 * и ISO-строки.
 */
import type { BadgeTone } from "../../design/primitives";

export type Role = "admin" | "teacher" | "rop" | "student";

export const ROLES: Role[] = ["admin", "teacher", "rop", "student"];

export const ROLE_LABEL: Record<Role, string> = {
  admin: "Админ",
  teacher: "Преподаватель",
  rop: "РОП",
  student: "Студент",
};

export const ROLE_TONE: Record<Role, BadgeTone> = {
  admin: "danger",
  teacher: "primary",
  rop: "info",
  student: "success",
};

function isRole(value: string): value is Role {
  return value === "admin" || value === "teacher" || value === "rop" || value === "student";
}

/** Приводит произвольную строку сервера к известной роли (иначе «student»). */
export function asRole(value: string): Role {
  return isRole(value) ? value : "student";
}

/** Человекочитаемая подпись роли; неизвестные значения показываются как есть. */
export function roleLabel(role: string): string {
  return isRole(role) ? ROLE_LABEL[role] : role;
}

/** Тон бейджа роли; неизвестные значения получают нейтральный тон. */
export function roleTone(role: string): BadgeTone {
  return isRole(role) ? ROLE_TONE[role] : "neutral";
}

/** Форматирует дату: unix-секунды, unix-мс или ISO строка -> локальная ru-RU строка. */
export function fmtDateTime(value: string | number | null | undefined): string {
  if (value === null || value === undefined || value === "") return "-";
  const raw = String(value).trim();
  let d: Date;
  if (/^\d+(\.\d+)?$/.test(raw)) {
    const n = Number(raw);
    d = new Date(n > 1e11 ? n : n * 1000);
  } else {
    d = new Date(raw);
  }
  return isNaN(d.getTime()) ? "-" : d.toLocaleString("ru-RU");
}
