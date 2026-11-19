/**
 * Константы вкладки «Пользователи» админ-консоли.
 *
 * Перенесено из AdminUsersTab.tsx без изменений значений.
 */
import type { ActionKind } from "./types";

export const DEACTIVATION_WORD = "УДАЛИТЬ";

export const SKELETON_ROWS = 6;
export const selectClass =
  "h-9 px-3 rounded-lg border border-input bg-input-background text-sm";

export const ACTION_META: Record<ActionKind, { title: string; confirm: string }> = {
  deactivate: { title: "Деактивация пользователя", confirm: "Деактивировать" },
  password: { title: "Сброс пароля", confirm: "Сохранить пароль" },
  revoke: { title: "Отзыв сессий", confirm: "Отозвать сессии" },
  twofa: { title: "Сброс 2FA", confirm: "Сбросить 2FA" },
  role: { title: "Смена роли", confirm: "Сохранить роль" },
};

/**
 * Действия, которые сервер выполняет только с подтверждением caller_password,
 * если цель - другой администратор. Отзыв сессий этого не требует.
 */
export const STEP_UP_KINDS: Record<ActionKind, boolean> = {
  deactivate: true,
  password: true,
  revoke: false,
  twofa: true,
  role: true,
};
