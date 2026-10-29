import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

interface PageHeaderProps {
  title: string;
  description?: string;
  icon?: LucideIcon;
  actions?: ReactNode;
}

/**
 * Заголовок раздела: иконка в нейтральном квадрате, заголовок, описание.
 * Раньше каждый экран дублировал «градиентную плашку» с ореолом и
 * градиентным текстом — визуальный шум вместо смысла.
 */
export function PageHeader({ title, description, icon: Icon, actions }: PageHeaderProps) {
  return (
    <div className="flex items-start gap-4 mb-6">
      {Icon && (
        <div className="shrink-0 grid place-items-center size-10 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300">
          <Icon className="size-5" />
        </div>
      )}
      <div className="min-w-0">
        <h1 className="text-2xl font-semibold tracking-tight text-slate-900 dark:text-slate-50">
          {title}
        </h1>
        {description && (
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400 max-w-2xl">
            {description}
          </p>
        )}
      </div>
      {actions && <div className="ml-auto shrink-0 flex items-center gap-2">{actions}</div>}
    </div>
  );
}
