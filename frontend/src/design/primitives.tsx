/**
 * Примитивы дизайн-системы (data-product, акцент #2563EB).
 *
 * Новые экраны собираются из этих компонентов; старые продолжают работать.
 * Все цвета берутся из CSS-переменных (`styles/theme.css`), поэтому светлая
 * и тёмная темы переключаются без правок в компонентах.
 */

import {
  type ButtonHTMLAttributes,
  type HTMLAttributes,
  type ReactNode,
} from "react";

import { cn } from "../app/components/ui/utils";

/* ------------------------------------------------------------------ Button */

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";
type ButtonSize = "sm" | "md";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
  iconLeft?: ReactNode;
  iconRight?: ReactNode;
}

const BUTTON_VARIANT: Record<ButtonVariant, string> = {
  primary:
    "bg-primary text-primary-foreground hover:brightness-110 active:brightness-95 shadow-xs",
  secondary:
    "bg-surface text-foreground border border-border hover:bg-muted active:bg-muted/70",
  ghost: "bg-transparent text-muted-foreground hover:bg-muted hover:text-foreground",
  danger:
    "bg-destructive text-destructive-foreground hover:brightness-110 shadow-xs",
};

const BUTTON_SIZE: Record<ButtonSize, string> = {
  sm: "h-8 px-3 text-xs gap-1.5 rounded-md",
  md: "h-10 px-4 text-sm gap-2 rounded-lg",
};

export function Button({
  variant = "primary",
  size = "md",
  loading = false,
  iconLeft,
  iconRight,
  className,
  children,
  disabled,
  ...rest
}: ButtonProps) {
  return (
    <button
      {...rest}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={cn(
        "inline-flex items-center justify-center font-medium transition-colors",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
        "disabled:opacity-50 disabled:pointer-events-none whitespace-nowrap",
        BUTTON_SIZE[size],
        BUTTON_VARIANT[variant],
        className,
      )}
    >
      {loading ? <Spinner /> : iconLeft}
      {children}
      {iconRight}
    </button>
  );
}

function Spinner() {
  return (
    <span
      aria-hidden="true"
      className="size-3.5 animate-spin rounded-full border-2 border-current border-t-transparent"
    />
  );
}

/* -------------------------------------------------------------------- Card */

type CardTone = "default" | "flat" | "interactive";

export interface CardProps extends HTMLAttributes<HTMLDivElement> {
  tone?: CardTone;
}

const CARD_TONE: Record<CardTone, string> = {
  default: "bg-card border border-border shadow-sm",
  flat: "bg-card border border-border",
  interactive:
    "bg-card border border-border shadow-sm transition-all hover:shadow-md hover:border-border-strong",
};

export function Card({ tone = "default", className, ...rest }: CardProps) {
  return <div {...rest} className={cn("rounded-lg", CARD_TONE[tone], className)} />;
}

export function CardHeader({
  title,
  description,
  actions,
  className,
}: {
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex items-start justify-between gap-4 px-5 pt-5 pb-3", className)}>
      <div className="min-w-0">
        <h3 className="text-base font-semibold text-foreground truncate">{title}</h3>
        {description ? (
          <p className="mt-0.5 text-xs text-muted-foreground">{description}</p>
        ) : null}
      </div>
      {actions ? <div className="flex shrink-0 items-center gap-2">{actions}</div> : null}
    </div>
  );
}

export function CardBody({ className, ...rest }: HTMLAttributes<HTMLDivElement>) {
  return <div {...rest} className={cn("px-5 pb-5", className)} />;
}

/* ---------------------------------------------------------------- PageHeader */

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
  className,
}: {
  eyebrow?: ReactNode;
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex flex-wrap items-end justify-between gap-4 mb-6", className)}>
      <div className="min-w-0">
        {eyebrow ? (
          <div className="text-[11px] font-semibold uppercase tracking-wider text-primary mb-1.5">
            {eyebrow}
          </div>
        ) : null}
        <h1 className="text-2xl font-semibold text-foreground tracking-tight">{title}</h1>
        {description ? (
          <p className="mt-1.5 max-w-2xl text-sm text-muted-foreground">{description}</p>
        ) : null}
      </div>
      {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
    </div>
  );
}

/* ------------------------------------------------------------------ Badge */

export type BadgeTone = "neutral" | "primary" | "success" | "warning" | "danger" | "info";

const BADGE_TONE: Record<BadgeTone, string> = {
  neutral: "bg-muted text-muted-foreground",
  primary: "bg-primary-soft text-primary-soft-foreground",
  success: "bg-success-soft text-success",
  warning: "bg-warning-soft text-warning",
  danger: "bg-destructive/10 text-destructive",
  info: "bg-info-soft text-info",
};

export function Badge({
  tone = "neutral",
  className,
  children,
  ...rest
}: HTMLAttributes<HTMLSpanElement> & { tone?: BadgeTone }) {
  return (
    <span
      {...rest}
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium",
        BADGE_TONE[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

/* ---------------------------------------------------------------- StatCard */

export function StatCard({
  label,
  value,
  hint,
  tone = "default",
  icon,
  className,
}: {
  label: ReactNode;
  value: ReactNode;
  hint?: ReactNode;
  tone?: "default" | "primary" | "success" | "warning" | "danger";
  icon?: ReactNode;
  className?: string;
}) {
  const accent: Record<string, string> = {
    default: "text-foreground",
    primary: "text-primary",
    success: "text-success",
    warning: "text-warning",
    danger: "text-destructive",
  };

  return (
    <Card className={cn("p-5", className)}>
      <div className="flex items-start justify-between gap-3">
        <div className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
          {label}
        </div>
        {icon ? <div className={cn("shrink-0", accent[tone])}>{icon}</div> : null}
      </div>
      <div className={cn("mt-2 text-2xl font-semibold tabular tracking-tight", accent[tone])}>
        {value}
      </div>
      {hint ? <div className="mt-1 text-xs text-muted-foreground">{hint}</div> : null}
    </Card>
  );
}

/* ------------------------------------------------------- Empty / Error states */

export function EmptyState({
  icon,
  title,
  description,
  action,
  className,
}: {
  icon?: ReactNode;
  title: ReactNode;
  description?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-border px-6 py-12 text-center",
        className,
      )}
    >
      {icon ? <div className="text-muted-foreground">{icon}</div> : null}
      <div className="text-sm font-medium text-foreground">{title}</div>
      {description ? (
        <p className="max-w-md text-xs text-muted-foreground">{description}</p>
      ) : null}
      {action}
    </div>
  );
}

export function ErrorState({
  title = "Не удалось загрузить данные",
  description,
  onRetry,
  className,
}: {
  title?: ReactNode;
  description?: ReactNode;
  onRetry?: () => void;
  className?: string;
}) {
  return (
    <div
      role="alert"
      className={cn(
        "flex flex-col items-center gap-3 rounded-lg border border-destructive/30 bg-destructive/5 px-6 py-10 text-center",
        className,
      )}
    >
      <div className="text-sm font-medium text-destructive">{title}</div>
      {description ? (
        <p className="max-w-md text-xs text-muted-foreground">{description}</p>
      ) : null}
      {onRetry ? (
        <Button variant="secondary" size="sm" onClick={onRetry}>
          Повторить
        </Button>
      ) : null}
    </div>
  );
}

/* ----------------------------------------------------------------- Skeleton */

export function Skeleton({ className }: { className?: string }) {
  return <div aria-hidden="true" className={cn("skeleton rounded-md", className)} />;
}

/* -------------------------------------------------------------- TableFrame */

export function TableFrame({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("overflow-x-auto rounded-lg border border-border bg-card", className)}>
      {children}
    </div>
  );
}

export const tableHeadClass =
  "bg-muted/60 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground";

export const tableRowClass =
  "border-b border-border last:border-0 hover:bg-muted/40 transition-colors";

/* ------------------------------------------------------------------- Field */

export function Field({
  label,
  hint,
  error,
  htmlFor,
  children,
  className,
}: {
  label: ReactNode;
  hint?: ReactNode;
  error?: ReactNode;
  htmlFor?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("space-y-1.5", className)}>
      <label htmlFor={htmlFor} className="block text-xs font-medium text-foreground">
        {label}
      </label>
      {children}
      {error ? (
        <p className="text-[11px] text-destructive">{error}</p>
      ) : hint ? (
        <p className="text-[11px] text-muted-foreground">{hint}</p>
      ) : null}
    </div>
  );
}

export const inputClass =
  "h-10 w-full rounded-lg border border-input bg-input-background px-3 text-sm text-foreground " +
  "placeholder:text-subtle-foreground transition-colors " +
  "focus-visible:border-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/30";

/* ------------------------------------------------------------------ Toolbar */

export function Toolbar({ className, ...rest }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      {...rest}
      className={cn("flex flex-wrap items-center gap-2 border-b border-border px-5 py-3", className)}
    />
  );
}

/* ------------------------------------------------------------------ Section */

export function SectionTitle({
  title,
  hint,
  actions,
  className,
}: {
  title: ReactNode;
  hint?: ReactNode;
  actions?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("mb-3 flex items-end justify-between gap-3", className)}>
      <div>
        <h2 className="text-[15px] font-semibold text-foreground">{title}</h2>
        {hint ? <p className="text-xs text-muted-foreground">{hint}</p> : null}
      </div>
      {actions}
    </div>
  );
}
