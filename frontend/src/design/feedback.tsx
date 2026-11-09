/**
 * Фидбек-примитивы: полоса прогресса и уведомляющие баннеры.
 *
 * Расширяет `primitives.tsx`, не дублируя его: там - каркасные блоки,
 * здесь - компоненты состояния (сколько выполнено, что произошло).
 * Все цвета - дизайн-токены из `styles/theme.css`, светлая/тёмная тема
 * переключаются без правок здесь.
 */

import { motion } from "motion/react";
import type { HTMLAttributes, ReactNode } from "react";

import { cn } from "../app/components/ui/utils";

/* ---------------------------------------------------------------- Progress */

export type ProgressTone = "primary" | "success" | "warning" | "danger";

const PROGRESS_FILL: Record<ProgressTone, string> = {
  primary: "bg-primary",
  success: "bg-success",
  warning: "bg-warning",
  danger: "bg-destructive",
};

export interface ProgressProps {
  /** 0..100; значения вне диапазона обрезаются. */
  value: number;
  /** Подпись для скринридеров, обязательна: без неё progressbar безымянный. */
  label: string;
  /** Тональность заливки - дизайн-токены. */
  tone?: ProgressTone;
  className?: string;
}

/**
 * Полоса прогресса с a11y-атрибутами. Ширина заливки анимируется через
 * motion (как в PipelineProgress), поэтому inline style не нужен.
 */
export function Progress({ value, label, tone = "primary", className }: ProgressProps) {
  const pct = Math.min(100, Math.max(0, Math.round(value)));
  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct}
      aria-valuetext={`${pct}%`}
      className={cn("relative h-2 w-full overflow-hidden rounded-full bg-muted", className)}
    >
      <motion.div
        className={cn("absolute inset-y-0 left-0 rounded-full", PROGRESS_FILL[tone])}
        initial={false}
        animate={{ width: `${pct}%` }}
        transition={{ duration: 0.4, ease: "easeOut" }}
      />
    </div>
  );
}

/* ----------------------------------------------------------------- Callout */

export type CalloutTone = "info" | "success" | "warning" | "danger";

const CALLOUT_TONE: Record<CalloutTone, string> = {
  info: "border-info/30 bg-info-soft",
  success: "border-success/30 bg-success-soft",
  warning: "border-warning/30 bg-warning-soft",
  danger: "border-destructive/30 bg-destructive/5",
};

const CALLOUT_TEXT: Record<CalloutTone, string> = {
  info: "text-info",
  success: "text-success",
  warning: "text-warning",
  danger: "text-destructive",
};

export interface CalloutProps extends HTMLAttributes<HTMLDivElement> {
  tone?: CalloutTone;
  /** Иконка слева; прячьте её от скринридеров (aria-hidden). */
  icon?: ReactNode;
  /** Заявленная цель баннера: «alert» перечитывается немедленно,
   * «status» - вежливое уведомление. По умолчанию «status». */
  role?: "alert" | "status";
  /** Действие справа (кнопка «Повторить» и т.п.). */
  action?: ReactNode;
}

/** Баннер-уведомление: тон задаётся токенами, роли - семантикой. */
export function Callout({
  tone = "info",
  icon,
  role = "status",
  action,
  className,
  children,
  ...rest
}: CalloutProps) {
  return (
    <div
      {...rest}
      role={role}
      className={cn(
        "flex flex-wrap items-center justify-between gap-2 rounded-lg border px-3 py-2 text-sm",
        CALLOUT_TONE[tone],
        className,
      )}
    >
      <span className={cn("flex min-w-0 items-start gap-2", CALLOUT_TEXT[tone])}>
        {icon ? <span className="mt-0.5 shrink-0">{icon}</span> : null}
        <span className="min-w-0">{children}</span>
      </span>
      {action ? <span className="shrink-0">{action}</span> : null}
    </div>
  );
}
