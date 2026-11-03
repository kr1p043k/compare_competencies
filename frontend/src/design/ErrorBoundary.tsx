import { Component, type ErrorInfo, type ReactNode } from "react";

import { reportError } from "../lib/errors";

type Props = {
  /** Заголовок по умолчанию, если вкладка не передала свой. */
  title?: string;
  /** Что именно упало: используется в тексте и в логе. */
  scope?: string;
  children: ReactNode;
};

type State = {
  error: Error | null;
};

/**
 * Граница падения для одной вкладки/панели.
 *
 * Раньше любое исключение в рендере (битый ответ API, неожидаемая форма
 * данных) размонтировало всё приложение: пользователь видел белый экран и
 * терял навигацию. Теперь падает только обёрнутый участок, остальное работает,
 * а пользователь видит понятное сообщение и кнопку «Повторить».
 *
 * Class-компонент - требование React: getDerivedStateFromError есть только
 * у классовых компонентов, хуков-аналога нет.
 */
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    reportError(
      `ErrorBoundary/${this.props.scope ?? "app"}`,
      error,
    );
    // Компонентный стек полезен для разбора: React не кладёт его в error.
    if (info.componentStack) {
      console.error(`[ErrorBoundary:${this.props.scope ?? "app"}]`, info.componentStack);
    }
  }

  private reset = () => {
    this.setState({ error: null });
  };

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    const title = this.props.title ?? "Раздел не загрузился";

    return (
      <div
        role="alert"
        className="flex flex-col items-center gap-3 rounded-lg border border-destructive/30 bg-destructive/5 px-6 py-10 text-center"
      >
        <div className="text-sm font-medium text-destructive">{title}</div>
        <p className="max-w-md text-xs text-muted-foreground">
          Произошла ошибка при отображении. Остальные разделы продолжают работать.
        </p>
        <p className="max-w-md break-words text-xs text-muted-foreground/80">
          {error.message || String(error)}
        </p>
        <button
          type="button"
          onClick={this.reset}
          className="rounded-md border border-border bg-card px-3 py-1.5 text-xs font-medium text-foreground transition-colors hover:bg-muted"
        >
          Повторить
        </button>
      </div>
    );
  }
}
