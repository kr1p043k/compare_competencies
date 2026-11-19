/**
 * Мелкие helper-функции вкладки «Пользователи».
 *
 * Перенесено из AdminUsersTab.tsx без изменений поведения.
 */
export function truncate(text: string, max = 24): string {
  if (!text) return "-";
  return text.length > max ? `${text.slice(0, max)}…` : text;
}
