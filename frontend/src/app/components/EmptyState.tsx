import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

import { EmptyState as DesignEmptyState } from "../../design";

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
    <DesignEmptyState
      icon={Icon ? <Icon className={compact ? "size-5" : "size-8"} /> : undefined}
      title={title}
      description={description}
      action={action ? <div className="mt-1">{action}</div> : undefined}
      className={compact ? "gap-1 px-3 py-6" : undefined}
    />
  );
}
