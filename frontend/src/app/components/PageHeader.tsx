import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

import { PageHeader as DesignPageHeader } from "../../design";

interface PageHeaderProps {
  title: string;
  description?: string;
  icon?: LucideIcon;
  actions?: ReactNode;
}

/**
 * Заголовок раздела: иконка в нейтральном квадрате, заголовок, описание.
 * Раньше каждый экран дублировал «градиентную плашку» с ореолом и
 * градиентным текстом - визуальный шум вместо смысла.
 */
export function PageHeader({ title, description, icon: Icon, actions }: PageHeaderProps) {
  return (
    <DesignPageHeader
      title={title}
      description={description}
      icon={Icon}
      actions={actions}
    />
  );
}
