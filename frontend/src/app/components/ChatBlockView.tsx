/**
 * Рендеры блоков ответа ассистента (компоненты-соседи LlmChatPanel).
 *
 * Контракт типов блоков - серверный `_ALLOWED_BLOCK_TYPES`
 * (src/services/llm_chat.py): "text" | "metric" | "citation" | "chart" |
 * "action_list". Точная форма полей у LLM не фиксирована (модель может
 * добавить/переименовать поля), поэтому каждый рендер достаёт данные
 * мягко: нет данных -> аккуратный фолбэк, никаких исключений и выдуманных
 * значений.
 */

import { lazy, Suspense, type ReactNode } from "react";
import { ExternalLink, Link2, ListOrdered } from "lucide-react";
import { Card, Skeleton, StatCard } from "../../design";
import { ChatMarkdown } from "./ChatMarkdown";
import { cn } from "./ui/utils";
// Чистое распознавание данных графика живёт отдельно от recharts-рендера:
// статический импорт не тянет за собой recharts, сам рендер ниже - ленивый.
import { extractChartData } from "./ChatChartBlock.data";

// recharts тяжёлый: грузим лениво, только когда реально пришёл chart-блок
// со распознаваемыми данными.
const ChatChartBlock = lazy(() =>
  import("./ChatChartBlock").then((m) => ({ default: m.ChatChartBlock })),
);

/** Структурный блок из SSE-события `block` или истории. */
export interface ChatBlock {
  type: string;
  [field: string]: unknown;
}

/* ---------------------------------------------------------------- утилиты */

function strOf(value: unknown): string {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "bigint") return String(value);
  return "";
}

/** Значение метрики: числа и строки показываем как есть, без выдуманных единиц. */
function formatValue(value: unknown): string {
  return strOf(value) || "—";
}

/** Json-событие `block` -> объект блока; мусор (не объект / без type) -> null. */
export function normalizeBlock(raw: unknown): ChatBlock | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const rec = raw as Record<string, unknown>;
  if (typeof rec.type !== "string" || rec.type === "") return null;
  return rec as ChatBlock;
}

/** Текстовое содержимое блока: сервер читает `content` или `text`
 * (см. `_blocks_text` в llm_chat.py). */
export function extractBlockText(block: ChatBlock): string {
  return strOf(block.content) || strOf(block.text);
}

/* ------------------------------------------------------------------ metric */

export interface MetricTile {
  label: string;
  value: string;
  hint?: string;
}

/** Блок может нести одну метрику ({"type":"metric","name":...,"value":...})
 * или список в metrics/items/values. Отсутствующих полей не изобретаем. */
export function extractMetrics(block: ChatBlock): MetricTile[] {
  const tilesFromEntry = (entry: unknown): MetricTile | null => {
    if (typeof entry === "string") {
      const t = entry.trim();
      return t ? { label: t, value: "—" } : null;
    }
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) return null;
    const rec = entry as Record<string, unknown>;
    const label = strOf(rec.name) || strOf(rec.label) || strOf(rec.title);
    const rawValue = rec.value ?? rec.score ?? rec.percent;
    if (!label && rawValue === undefined) return null;
    const hint = strOf(rec.hint) || strOf(rec.description) || strOf(rec.note);
    return {
      label: label || "Метрика",
      value: rawValue === undefined ? "—" : formatValue(rawValue),
      hint: hint || undefined,
    };
  };

  for (const key of ["metrics", "items", "values"]) {
    const group = block[key];
    if (!Array.isArray(group) || group.length === 0) continue;
    const tiles = group.map(tilesFromEntry).filter((t): t is MetricTile => t !== null);
    if (tiles.length > 0) return tiles;
  }
  const single = tilesFromEntry(block);
  return single ? [single] : [];
}

function MetricBlock({ tiles }: { tiles: MetricTile[] }) {
  return (
    <div className="grid grid-cols-2 gap-2">
      {tiles.map((tile, i) => (
        <StatCard
          key={i}
          label={tile.label}
          value={tile.value}
          hint={tile.hint}
          className={cn("p-4 text-left", tiles.length === 1 && "col-span-2")}
        />
      ))}
    </div>
  );
}

/* ---------------------------------------------------------------- citation */

export interface CitationItem {
  title: string;
  url: string | null;
}

/** Внешняя ссылка должна вести на источник: http(s) или относительный путь. */
function sanitizeUrl(raw: string): string | null {
  if (/^https?:\/\//i.test(raw)) return raw;
  if (raw.startsWith("/") && !raw.startsWith("//")) return raw;
  return null;
}

export function extractCitations(block: ChatBlock): CitationItem[] {
  const itemFromEntry = (entry: unknown): CitationItem | null => {
    if (typeof entry === "string") {
      const t = entry.trim();
      return t ? { title: t, url: sanitizeUrl(t) } : null;
    }
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) return null;
    const rec = entry as Record<string, unknown>;
    const title = strOf(rec.title) || strOf(rec.name) || strOf(rec.label) ||
      strOf(rec.content) || strOf(rec.text);
    const rawUrl = strOf(rec.url) || strOf(rec.link) || strOf(rec.href) || strOf(rec.source);
    const url = sanitizeUrl(rawUrl);
    if (!title && !url) return null;
    return { title: title || (url as string), url };
  };

  for (const key of ["citations", "sources", "items", "links"]) {
    const group = block[key];
    if (!Array.isArray(group) || group.length === 0) continue;
    const items = group.map(itemFromEntry).filter((c): c is CitationItem => c !== null);
    if (items.length > 0) return items;
  }
  const single = itemFromEntry(block);
  return single ? [single] : [];
}

function CitationBlock({ items }: { items: CitationItem[] }) {
  return (
    <Card tone="flat" className="p-3">
      <ul className="space-y-1.5 text-sm">
        {items.map((item, i) =>
          item.url ? (
            <li key={i} className="flex items-start gap-1.5">
              <Link2 className="mt-0.5 size-3.5 shrink-0 text-muted-foreground" aria-hidden="true" />
              <a
                href={item.url}
                target={item.url.startsWith("http") ? "_blank" : undefined}
                rel={item.url.startsWith("http") ? "noopener noreferrer" : undefined}
                className="break-all text-primary underline underline-offset-2"
              >
                {item.title}
                <ExternalLink className="ml-1 inline size-3 text-muted-foreground" aria-hidden="true" />
              </a>
            </li>
          ) : (
            <li key={i} className="flex items-start gap-1.5 text-foreground">
              <Link2 className="mt-0.5 size-3.5 shrink-0 text-muted-foreground" aria-hidden="true" />
              <span>{item.title}</span>
            </li>
          ),
        )}
      </ul>
    </Card>
  );
}

/* --------------------------------------------------------------- action_list */

export interface ActionItem {
  text: string;
  note?: string;
}

export function extractActions(block: ChatBlock): ActionItem[] {
  const itemFromEntry = (entry: unknown): ActionItem | null => {
    if (typeof entry === "string") {
      const t = entry.trim();
      return t ? { text: t } : null;
    }
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) return null;
    const rec = entry as Record<string, unknown>;
    const text = strOf(rec.text) || strOf(rec.title) || strOf(rec.content) ||
      strOf(rec.description) || strOf(rec.suggestion);
    if (!text) return null;
    const note = strOf(rec.note) || strOf(rec.priority);
    return { text, note: note || undefined };
  };

  for (const key of ["items", "actions", "steps", "points"]) {
    const group = block[key];
    if (!Array.isArray(group) || group.length === 0) continue;
    const items = group.map(itemFromEntry).filter((a): a is ActionItem => a !== null);
    if (items.length > 0) return items;
  }
  const single = itemFromEntry(block);
  return single ? [single] : [];
}

function ActionListBlock({ block }: { block: ChatBlock }) {
  const items = extractActions(block);
  const title = strOf(block.title) || strOf(block.heading);
  if (items.length === 0) {
    const text = extractBlockText(block);
    return text ? <ChatMarkdown>{text}</ChatMarkdown> : null;
  }
  return (
    <Card tone="flat" className="p-3">
      {title && (
        <div className="mb-1.5 flex items-center gap-1.5 text-xs font-semibold text-foreground">
          <ListOrdered className="size-3.5 text-muted-foreground" aria-hidden="true" />
          {title}
        </div>
      )}
      <ol className="list-decimal space-y-1.5 pl-5 text-sm">
        {items.map((item, i) => (
          <li key={i}>
            {item.text}
            {item.note && <span className="ml-1 text-xs text-muted-foreground">({item.note})</span>}
          </li>
        ))}
      </ol>
    </Card>
  );
}

/* ------------------------------------------------------------------ фолбэки */

/** Данные блока не распознаны: показываем честный фолбэк, ничего не теряем. */
function FallbackBlock({
  block,
  note,
}: {
  block: ChatBlock;
  note?: string;
}) {
  let dump: string;
  try {
    dump = JSON.stringify(block, null, 2);
  } catch {
    dump = String(block);
  }
  return (
    <details className="rounded-lg border border-dashed border-border bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
      <summary className="cursor-pointer font-medium text-foreground">
        {note ?? `Блок «${block.type}» без рендера`}
      </summary>
      <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap break-all font-mono">{dump}</pre>
    </details>
  );
}

/* -------------------------------------------------------------------- роутер */

export function ChatBlockView({ block }: { block: ChatBlock }): ReactNode {
  switch (block.type) {
    case "text": {
      const text = extractBlockText(block);
      return text ? <ChatMarkdown>{text}</ChatMarkdown> : null;
    }

    case "metric": {
      const tiles = extractMetrics(block);
      if (tiles.length === 0) return <FallbackBlock block={block} note="Метрика без данных" />;
      return <MetricBlock tiles={tiles} />;
    }

    case "citation": {
      const items = extractCitations(block);
      if (items.length === 0) {
        const text = extractBlockText(block);
        return text ? <ChatMarkdown>{text}</ChatMarkdown> : null;
      }
      return <CitationBlock items={items} />;
    }

    case "chart": {
      // Распознавание данных - чистая функция (тестируется отдельно);
      // без данных recharts не грузим.
      const data = extractChartData(block);
      if (!data) {
        const text = extractBlockText(block);
        if (text) return <ChatMarkdown>{text}</ChatMarkdown>;
        return <FallbackBlock block={block} note="Диаграмма без распознаваемых данных" />;
      }
      return (
        <Suspense fallback={<Skeleton className="h-44 w-full" />}>
          <ChatChartBlock data={data} />
        </Suspense>
      );
    }

    case "action_list":
      return <ActionListBlock block={block} />;

    default:
      return <FallbackBlock block={block} />;
  }
}
