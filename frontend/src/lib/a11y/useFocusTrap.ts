import { useEffect, useRef, type RefObject } from "react";

/** Фокусируемые элементы внутри контейнера (видимые, в порядке DOM). */
const FOCUSABLE_SELECTOR = [
  "a[href]",
  "button:not([disabled])",
  "textarea:not([disabled])",
  "input:not([disabled])",
  'select:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
  '[contenteditable="true"]',
].join(",");

function focusable(container: HTMLElement): HTMLElement[] {
  return Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)).filter(
    (el) => !el.hasAttribute("hidden") && el.getAttribute("aria-hidden") !== "true",
  );
}

/**
 * Ловушка фокуса для модальных окон и дроверов, сделанных вручную (не Radix).
 *
 * Пока `active`, фокус не выходит за пределы контейнера по Tab/Shift+Tab, а
 * Escape вызывает `onClose`. При закрытии фокус возвращается туда, где был до
 * открытия - иначе после закрытия дровера клавиатурный пользователь оказывается
 * в начале страницы. Radix Dialog это делает сам, кастомные оверлеи - нет.
 */
export function useFocusTrap<T extends HTMLElement>(
  active: boolean,
  onClose?: () => void,
): RefObject<T> {
  const ref = useRef<T>(null);
  // onClose держим в ref: иначе инлайн-колбэк меняет идентичность каждый рендер,
  // эффект перезапускается и на каждой отрисовке заново переводит фокус.
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    if (!active) return;
    const node = ref.current;
    if (!node) return;

    const previouslyFocused = document.activeElement as HTMLElement | null;

    // Начальный фокус: первый доступный элемент, иначе сам контейнер.
    const initial = focusable(node)[0];
    (initial ?? node).focus();

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        if (onCloseRef.current) {
          event.stopPropagation();
          onCloseRef.current();
        }
        return;
      }
      if (event.key !== "Tab") return;

      const items = focusable(node);
      if (items.length === 0) {
        // Нечего фокусировать - держим фокус на контейнере.
        event.preventDefault();
        node.focus();
        return;
      }
      const first = items[0];
      const last = items[items.length - 1];
      const current = document.activeElement as HTMLElement | null;

      if (event.shiftKey && (current === first || !node.contains(current))) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && current === last) {
        event.preventDefault();
        first.focus();
      }
    };

    node.addEventListener("keydown", onKeyDown);
    return () => {
      node.removeEventListener("keydown", onKeyDown);
      previouslyFocused?.focus?.();
    };
  }, [active]);

  return ref;
}
