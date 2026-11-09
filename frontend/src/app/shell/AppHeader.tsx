import { useMemo } from "react";
import {
  ChevronDown,
  GraduationCap,
  LogOut,
  Moon,
  Shield,
  Sun,
  TrendingUp,
  User,
  UserCheck,
} from "lucide-react";

import { Badge, Button, inputClass } from "../../design";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "../components/ui/dropdown-menu";
import { TabsList } from "../components/ui/tabs";
import { cn } from "../components/ui/utils";
import type { Theme } from "../../lib/theme";

import { buildNavGroups, type NavGroupDef, type NavItem } from "./tabConfig";

/**
 * Шапка приложения: логотип, имя пользователя и роль (реальная, не предпросмотр),
 * переход в профиль, селект предпросмотра роли (только admin), переключатель
 * темы, выход. Ниже шапки - баннер активного предпросмотра.
 *
 * Разрез из App.tsx: порядок и подписи сохранены дословно, все краски -
 * дизайн-токены из src/design (светлая и тёмная темы без правок здесь).
 */
export function AppHeader({
  name,
  role,
  canPreview,
  rolePreview,
  onPreviewChange,
  previewActive,
  onResetPreview,
  theme,
  onToggleTheme,
  onOpenProfile,
  onLogout,
}: {
  name: string | null;
  role: string | null;
  canPreview: boolean;
  rolePreview: string | null;
  onPreviewChange: (v: string | null) => void;
  previewActive: boolean;
  onResetPreview: () => void;
  theme: Theme;
  onToggleTheme: () => void;
  onOpenProfile: () => void;
  onLogout: () => void;
}) {
  const roleIcon = role === "admin" ? <Shield className="size-4" /> : (role === "teacher" || role === "rop") ? <UserCheck className="size-4" /> : <GraduationCap className="size-4" />;
  const roleLabel = role === "admin" ? "Администратор" : role === "teacher" ? "Преподаватель" : role === "rop" ? "Руководитель ОП" : role === "student" ? "Студент" : "-";

  return (
    <>
      {/* Header */}
      <header className="sticky top-0 z-40 border-b border-border bg-surface/90 backdrop-blur-xl">
        <div className="max-w-[1760px] mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className="flex items-center justify-center size-11 rounded-xl bg-primary shadow-md shadow-primary/25">
                <TrendingUp className="size-6 text-primary-foreground" aria-hidden="true" />
              </div>
              <div>
                <h1 className="text-xl font-semibold tracking-tight text-foreground">
                  Карта компетенций
                </h1>
                <p className="text-xs text-muted-foreground">
                  Сопоставление учебных программ с требованиями рынка труда
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <span aria-hidden="true">{roleIcon}</span>
                <span>{name || roleLabel}</span>
                <Badge tone="primary" className="text-xs">{roleLabel}</Badge>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={onOpenProfile}
                title="Мой профиль"
                className="cursor-pointer transition-colors duration-200"
              >
                <User className="size-4" aria-hidden="true" />
              </Button>
              {canPreview && (
                <label className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  Просмотр как
                  <select
                    value={rolePreview ?? ""}
                    onChange={(e) => onPreviewChange(e.target.value || null)}
                    className={cn(inputClass, "h-8 w-auto px-1.5 text-xs")}
                  >
                    <option value="">своя роль</option>
                    <option value="admin">admin</option>
                    <option value="teacher">teacher</option>
                    <option value="rop">rop</option>
                    <option value="student">student</option>
                  </select>
                </label>
              )}
              <Button
                variant="ghost"
                size="sm"
                onClick={onToggleTheme}
                title={theme === "dark" ? "Светлая тема" : "Тёмная тема"}
              >
                {theme === "dark" ? <Sun className="size-4" aria-hidden="true" /> : <Moon className="size-4" aria-hidden="true" />}
              </Button>
              <Button variant="ghost" size="sm" onClick={onLogout} className="hover:text-destructive">
                <LogOut className="size-4" aria-hidden="true" />
              </Button>
            </div>
          </div>
        </div>
      </header>
      {previewActive && (
        <div className="bg-warning-soft border-b border-warning/40">
          <div className="max-w-[1760px] mx-auto px-4 sm:px-6 lg:px-8 py-2 flex items-center justify-between gap-3">
            <p className="text-xs text-warning">
              Предпросмотр интерфейса роли «{rolePreview}». API-права не меняются - данные чужих ролей могут быть недоступны.
            </p>
            <Button variant="ghost" size="sm" onClick={onResetPreview} className="h-7 text-xs text-warning hover:text-warning">
              Сбросить
            </Button>
          </div>
        </div>
      )}
    </>
  );
}

/**
 * Локальный выпадающий список навигации: один пункт - кнопка,
 * несколько - DropdownMenu с активным пунктом на месте заголовка.
 */
function NavGroup({
  title,
  items,
  activeTab,
  onSelect,
}: {
  title: string;
  items: NavItem[];
  activeTab: string;
  onSelect: (v: string) => void;
}) {
  if (items.length === 0) return null;
  const active = items.find((i) => i.value === activeTab);
  if (items.length === 1) {
    const only = items[0];
    const isActive = only.value === activeTab;
    return (
      <button
        onClick={() => onSelect(only.value)}
        className={cn(
          "inline-flex items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-all cursor-pointer",
          isActive
            ? "bg-card text-foreground shadow-sm dark:bg-input/30"
            : "text-muted-foreground hover:text-foreground",
        )}
      >
        <only.Icon className="size-4" aria-hidden="true" />
        {only.label}
      </button>
    );
  }
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          className={cn(
            "inline-flex items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-all cursor-pointer",
            active
              ? "bg-card text-foreground shadow-sm dark:bg-input/30"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          {active ? <active.Icon className="size-4" aria-hidden="true" /> : null}
          {active ? active.label : title}
          <ChevronDown className="size-3.5 opacity-60" aria-hidden="true" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="min-w-52">
        {items.map(({ value, label, Icon }) => (
          <DropdownMenuItem
            key={value}
            onSelect={() => onSelect(value)}
            className={cn("gap-2 cursor-pointer", value === activeTab && "font-semibold text-primary")}
          >
            <Icon className="size-4" aria-hidden="true" />
            {label}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

/** Строка вкладок: три группы (Работа / Анализ / Система) из tabConfig. */
export function AppNavTabs({
  activeTab,
  effectiveRole,
  onSelect,
}: {
  activeTab: string;
  effectiveRole: string | null;
  onSelect: (v: string) => void;
}) {
  const groups = useMemo<NavGroupDef[]>(() => buildNavGroups(effectiveRole), [effectiveRole]);
  return (
    <TabsList className="inline-flex h-11 items-center justify-center gap-1 rounded-xl border border-border bg-muted/70 p-1 shadow-xs">
      {groups.map((g) => (
        <NavGroup key={g.title} title={g.title} items={g.items} activeTab={activeTab} onSelect={onSelect} />
      ))}
    </TabsList>
  );
}
