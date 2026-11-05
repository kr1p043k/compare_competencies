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

import { Button } from "../components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "../components/ui/dropdown-menu";
import { TabsList } from "../components/ui/tabs";
import type { Theme } from "../../lib/theme";

import { buildNavGroups, type NavGroupDef, type NavItem } from "./tabConfig";

/**
 * Шапка приложения: логотип, имя пользователя и роль (реальная, не предпросмотр),
 * переход в профиль, селект предпросмотра роли (только admin), переключатель
 * темы, выход. Ниже шапки - баннер активного предпросмотра.
 *
 * Разрез из App.tsx без изменения верстки: классы, порядок и подписи
 * сохранены дословно.
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
                <TrendingUp className="size-6 text-white" />
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
              <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-slate-400">
                {roleIcon}
                <span>{name || roleLabel}</span>
                <span className="text-xs px-2 py-0.5 rounded-full bg-primary-soft text-primary-soft-foreground font-medium">{roleLabel}</span>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={onOpenProfile}
                title="Мой профиль"
                className="cursor-pointer transition-colors duration-200 text-gray-500 hover:text-gray-900 dark:text-slate-400 dark:hover:text-slate-100 focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:outline-none"
              >
                <User className="size-4" />
              </Button>
              {canPreview && (
                <label className="flex items-center gap-1.5 text-xs text-gray-500 dark:text-slate-400">
                  Просмотр как
                  <select
                    value={rolePreview ?? ""}
                    onChange={(e) => onPreviewChange(e.target.value || null)}
                    className="h-8 rounded-md border border-gray-300 dark:border-slate-600 bg-white dark:bg-slate-900 px-1.5 text-xs text-gray-700 dark:text-slate-200"
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
                className="text-gray-500 hover:text-gray-900 dark:text-slate-400 dark:hover:text-slate-100"
              >
                {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
              </Button>
              <Button variant="ghost" size="sm" onClick={onLogout} className="text-gray-500 dark:text-slate-400 hover:text-red-600">
                <LogOut className="size-4" />
              </Button>
            </div>
          </div>
        </div>
      </header>
      {previewActive && (
        <div className="bg-amber-100 dark:bg-amber-950/40 border-b border-amber-300 dark:border-amber-800">
          <div className="max-w-[1760px] mx-auto px-4 sm:px-6 lg:px-8 py-2 flex items-center justify-between gap-3">
            <p className="text-xs text-amber-800 dark:text-amber-200">
              Предпросмотр интерфейса роли «{rolePreview}». API-права не меняются - данные чужих ролей могут быть недоступны.
            </p>
            <Button variant="ghost" size="sm" onClick={onResetPreview} className="h-7 text-xs text-amber-800 dark:text-amber-200">
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
        className={`inline-flex items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-all cursor-pointer ${
          isActive
            ? "bg-white text-gray-900 shadow-sm dark:bg-slate-800 dark:text-slate-100"
            : "text-gray-600 dark:text-slate-400 hover:text-gray-900 dark:hover:text-slate-200"
        }`}
      >
        <only.Icon className="size-4" />
        {only.label}
      </button>
    );
  }
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          className={`inline-flex items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-all cursor-pointer ${
            active
              ? "bg-white text-gray-900 shadow-sm dark:bg-slate-800 dark:text-slate-100"
              : "text-gray-600 dark:text-slate-400 hover:text-gray-900 dark:hover:text-slate-200"
          }`}
        >
          {active ? <active.Icon className="size-4" /> : null}
          {active ? active.label : title}
          <ChevronDown className="size-3.5 opacity-60" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="min-w-52">
        {items.map(({ value, label, Icon }) => (
          <DropdownMenuItem
            key={value}
            onSelect={() => onSelect(value)}
            className={`gap-2 cursor-pointer ${value === activeTab ? "font-semibold text-blue-700 dark:text-blue-300" : ""}`}
          >
            <Icon className="size-4" />
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
