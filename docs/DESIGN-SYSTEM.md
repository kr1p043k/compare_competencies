# Дизайн-система (data-product, акцент #2563EB)

Согласовано 2026-11-02: современный data-продукт, синий акцент, слой рядом с
существующим кодом (`src/design/*`), постепенная миграция экранов.

## Где что лежит

| Файл | Назначение |
|---|---|
| `src/styles/theme.css` | CSS-переменные светлой/тёмной темы, типографика, радиусы, тени |
| `src/styles/globals.css` | Скроллбар, фокус, скелетоны, reduced-motion, печать |
| `src/design/primitives.tsx` | Button, Card, PageHeader, StatCard, Badge, EmptyState, ErrorState, Skeleton, TableFrame, Field, Toolbar, SectionTitle |
| `src/design/chartTheme.ts` | Единые пропсы Recharts (оси, сетка, тултип, легенда) |
| `src/design/tokens.ts` | Палитра графиков, цвет тренда, тон бейджа достоверности |

## Токены

Поверхности: `background` (страница), `surface` (шапка/панели), `card`.
Текст: `foreground`, `muted-foreground`, `subtle-foreground`.
Акцент: `primary` (#2563EB), `primary-soft` (фон бейджей/ховеров).
Статусы: `success`, `warning`, `destructive`, `info` + `*-soft` варианты.
Радиусы: `xs 6 · sm 8 · md 12 · lg 14 · xl 18 · 2xl 24`.
Тени: `xs · sm · md · lg` + `shadow-focus`.
Графики: `chart-1…5` (синий, небесный, изумруд, янтарь, фиолет) и `chart-grid`.

Всё это доступно как Tailwind-классы: `bg-card`, `text-muted-foreground`,
`border-border`, `bg-primary-soft`, `shadow-md`, `rounded-lg`.

## Правила

1. Цвета — только из токенов. Никаких `text-gray-500` / `bg-slate-900` в новом коде.
2. Отступы — сетка 4 px: `p-4`, `gap-3`, `space-y-2`.
3. Одна метрика — один `StatCard`. Заголовок экрана — `PageHeader`.
4. Любая загрузка — `Skeleton`, любая ошибка — `ErrorState` с «Повторить»,
   любое пустое состояние — `EmptyState` с подсказкой, что делать.
5. Цифры в таблицах и метриках — класс `tabular` (не «прыгают»).
6. Фокус с клавиатуры всегда виден: `focus-visible:ring-2`.
7. Тёмная тема обязательна: проверять на ней каждый экран.
8. Recharts: брать `axisProps`, `gridProps`, `tooltipProps`, `legendProps`
   из `design/chartTheme.ts`, а цвета — из `design/tokens.ts`.

## Пример

```tsx
import { PageHeader, Card, StatCard, Badge, EmptyState } from "../../design";

<PageHeader
  eyebrow="Аналитика"
  title="Gap-анализ"
  description="Разрыв между программой и требованиями рынка"
  actions={<Button>Обновить</Button>}
/>

<div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
  <StatCard label="Компетенций" value="254" hint="из 312 в плане" tone="primary" />
</div>
```

## Состояния экрана (чек-лист)

- загрузка: скелетон повторяет форму контента, а не спиннер по центру;
- пусто: объяснение + действие («Запустить сбор», «Загрузить файл»);
- ошибка: текст по-русски + «Повторить», без технического стека;
- нет прав: отдельный текст, не 403-заглушка браузера;
- мобилка: одна колонка, таблицы горизонтально скроллятся в `TableFrame`.

## Миграция

Экраны переводятся по одному: сначала шапка и сетка (`PageHeader` + `Card`),
затем состояния, затем таблицы и графики. Старые классы Tailwind продолжают
работать — жёсткой миграции нет.

## Состав модулей `src/design`

| Модуль | Экспорт |
|---|---|
| `primitives.tsx` | Button, Card, CardHeader, CardBody, PageHeader, Badge, StatCard, MetricPair, EmptyState, ErrorState, Skeleton, TableFrame, Field, Toolbar, SectionTitle, Meter, Sparkbars, tableHeadClass, tableRowClass, inputClass |
| `feedback.tsx` | Progress, Callout |
| `ErrorBoundary.tsx` | ErrorBoundary (обёртка вкладки) |
| `chartTheme.ts` | axisProps, gridProps, tooltipProps, legendProps |
| `tokens.ts` | токены статусов, уровней, текста, кнопок |
| `index.ts` | реэкспорт всего выше |
