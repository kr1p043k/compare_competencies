import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

interface EmptyStateProps {
  icon?: LucideIcon;
  title: string;
  description?: string;
  action?: ReactNode;
  /** compact - для узких колонок и сайдбаров. */
  size?: "default" | "compact";
}

/**
 * Единое состояние «пусто» для списков и таблиц.
 *
 * Раньше в каждом компоненте был свой текст и свои отступы
 * («Нет данных», «Ничего не найдено», «Записей не найдено», «пусто»),
 * из-за чего пустой экран выглядел по-разному в соседних вкладках.
 */
export function EmptyState({ icon: Icon, title, description, action, size = "default" }: EmptyStateProps) {
  const compact = size === "compact";
  return (
    <div
      className={
        compact
          ? "flex flex-col items-center gap-1 px-3 py-6 text-center"
          : "flex flex-col items-center gap-2 px-4 py-12 text-center"
      }
    >
      {Icon && (
        <Icon className={compact ? "size-5 text-subtle-foreground" : "size-8 text-subtle-foreground"} />
      )}
      <p
        className={
          compact
            ? "text-xs text-muted-foreground"
            : "text-sm font-medium text-foreground"
        }
      >
        {title}
      </p>
      {description && (
        <p className="max-w-sm text-xs text-subtle-foreground">{description}</p>
      )}
      {action && <div className="mt-1">{action}</div>}
    </div>
  );
}
