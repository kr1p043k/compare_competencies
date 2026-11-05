/**
 * Конфигурация вкладок навигации: id, подпись, иконка и права доступа по ролям.
 *
 * Единственный источник правды о вкладках для трёх потребителей:
 *  - AppHeader (AppNavTabs) строит группы по эффективной роли;
 *  - useTabNavigation уводит с вкладки, недоступной роли, на безопасную;
 *  - App.tsx гейтит рендер TabsContent через isTabAllowedForRole.
 *
 * Сами известные значения вкладок (deep links `#/teacher`, нормализация
 * неизвестного хеша) остаются в lib/router (KNOWN_TABS).
 */
import type { LucideIcon } from "lucide-react";
import {
  Activity,
  BarChart3,
  BookOpen,
  Briefcase,
  Database,
  FileText,
  FolderOpen,
  GraduationCap,
  HelpCircle,
  History,
  LineChart,
  Shield,
  TrendingUp,
  User,
} from "lucide-react";

/** Пункт навигации: значение вкладки (id), подпись и иконка. */
export interface NavItem {
  value: string;
  label: string;
  Icon: LucideIcon;
}

/** Группа выпадающего меню в навигации. */
export interface NavGroupDef {
  title: string;
  items: NavItem[];
}

const isAdmin = (role: string | null) => role === "admin";
const isTeacherSide = (role: string | null) =>
  role === "teacher" || role === "rop" || role === "admin";

// Вкладки, повторяющиеся в разных группах / у разных ролей.
const RESULTS: NavItem = { value: "data", label: "Результаты", Icon: Database };
const VISUALIZATION: NavItem = { value: "visualization", label: "Визуализация", Icon: BarChart3 };
const TEACHER_ANALYSIS: NavItem = { value: "teacher", label: "Преподавательский анализ", Icon: BarChart3 };
const STUDENTS: NavItem = { value: "students", label: "Студенты", Icon: GraduationCap };
const MONITORING: NavItem = { value: "monitoring", label: "Мониторинг", Icon: Activity };
const LOGS: NavItem = { value: "logs", label: "Логи", Icon: FileText };
const ADMIN: NavItem = { value: "admin", label: "Админ", Icon: Shield };
const MY_REQUESTS: NavItem = { value: "student", label: "Мои запросы", Icon: History };
const TAXONOMY: NavItem = { value: "taxonomy", label: "Таксономия", Icon: BookOpen };

/**
 * Доступна ли вкладка роли. Переносит прежние списки App.tsx
 * (adminOnly / teacherOnly / studentOnly) в одно место.
 */
export function isTabAllowedForRole(tab: string, role: string | null): boolean {
  switch (tab) {
    case "visualization":
    case "monitoring":
    case "logs":
    case "admin":
      return isAdmin(role);
    case "taxonomy":
    case "teacher":
    case "students":
      return isTeacherSide(role);
    case "student":
      return role === "student" || isAdmin(role);
    default:
      return true;
  }
}

/**
 * Группы навигации для эффективной роли (с учётом предпросмотра админом).
 * Порядок пунктов сохранён прежним: слева направо в каждой группе.
 * Вкладка «Результаты» у студента переезжает в группу «Анализ».
 */
export function buildNavGroups(role: string | null): NavGroupDef[] {
  const admin = isAdmin(role);
  const teacherSide = isTeacherSide(role);

  return [
    {
      title: "Работа",
      items: [
        { value: "vacancies", label: "Вакансии", Icon: Briefcase },
        ...(role !== "student" ? [RESULTS] : []),
        ...(admin ? [VISUALIZATION] : []),
      ],
    },
    {
      title: "Анализ",
      items: [
        { value: "predictions", label: "Прогнозы", Icon: TrendingUp },
        { value: "articles", label: "Аналитика рынка", Icon: LineChart },
        { value: "scientific-trends", label: "Научные тренды", Icon: FolderOpen },
        { value: "publications", label: "Публикации ЮФУ", Icon: BookOpen },
        ...(role === "student" ? [RESULTS] : []),
        ...(teacherSide ? [TEACHER_ANALYSIS] : []),
        ...(teacherSide ? [STUDENTS] : []),
      ],
    },
    {
      title: "Система",
      items: [
        ...(admin ? [MONITORING, LOGS, ADMIN] : []),
        ...((role === "student" || admin) ? [MY_REQUESTS] : []),
        ...(teacherSide ? [TAXONOMY] : []),
        { value: "profile", label: "Мой профиль", Icon: User },
        { value: "help", label: "Помощь", Icon: HelpCircle },
      ],
    },
  ];
}
