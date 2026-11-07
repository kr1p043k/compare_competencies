/**
 * Минимальный безопасный markdown-рендер для чата с LLM.
 *
 * В проекте нет markdown-зависимостей (react-markdown и т.п.), а текст
 * ассистента приходит форматированным. Поддерживается компактный поднабор:
 * заголовки `#..###`, списки `- / * / 1.`, горизонтальная линия, код в
 * `` `...` `` и ```-блоках, **жирный**, *курсив*, [ссылки](https://...).
 *
 * Безопасность: никакого dangerouslySetInnerHTML - HTML из текста модели
 * НЕ интерпретируется, а показывается как текст (React экранирует сам).
 * Ссылки допускаются только http(s)/относительные; javascript: - нет.
 */

import { type ReactNode } from "react";
import { cn } from "./ui/utils";

/* --------------------------------------------------------------- разбор строк */

type Segment =
  | { kind: "code"; code: string }
  | { kind: "heading"; text: string }
  | { kind: "ul"; items: string[] }
  | { kind: "ol"; items: string[] }
  | { kind: "hr" }
  | { kind: "p"; text: string };

const FENCE_RE = /^```/;

function parseSegments(src: string): Segment[] {
  const segments: Segment[] = [];
  const lines = src.replace(/\r\n/g, "\n").split("\n");

  let paragraph: string[] = [];
  let list: { kind: "ul" | "ol"; items: string[] } | null = null;
  let code: string[] | null = null;

  const flushParagraph = () => {
    if (paragraph.length > 0) {
      segments.push({ kind: "p", text: paragraph.join(" ") });
      paragraph = [];
    }
  };
  const flushList = () => {
    if (list) {
      segments.push(list);
      list = null;
    }
  };

  for (const rawLine of lines) {
    const line = rawLine.trimEnd();

    if (code !== null) {
      if (FENCE_RE.test(line.trim())) {
        segments.push({ kind: "code", code: code.join("\n") });
        code = null;
      } else {
        code.push(rawLine);
      }
      continue;
    }

    if (FENCE_RE.test(line.trim())) {
      flushParagraph();
      flushList();
      code = [];
      continue;
    }

    const heading = /^(#{1,4})\s+(.*)$/.exec(line);
    if (heading) {
      flushParagraph();
      flushList();
      segments.push({ kind: "heading", text: heading[2] });
      continue;
    }

    if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
      flushParagraph();
      flushList();
      segments.push({ kind: "hr" });
      continue;
    }

    const ul = /^\s*[-*+]\s+(.*)$/.exec(line);
    if (ul) {
      flushParagraph();
      if (!list || list.kind !== "ul") {
        flushList();
        list = { kind: "ul", items: [] };
      }
      list.items.push(ul[1]);
      continue;
    }

    const ol = /^\s*\d+[.)]\s+(.*)$/.exec(line);
    if (ol) {
      flushParagraph();
      if (!list || list.kind !== "ol") {
        flushList();
        list = { kind: "ol", items: [] };
      }
      list.items.push(ol[1]);
      continue;
    }

    if (line.trim() === "") {
      flushParagraph();
      flushList();
      continue;
    }

    list && flushList();
    paragraph.push(line.trim());
  }

  if (code !== null) {
    // незакрытый блок кода - показываем накопленное как есть
    segments.push({ kind: "code", code: code.join("\n") });
  }
  flushParagraph();
  flushList();

  return segments;
}

/* ---------------------------------------------------------------- инлайновые */

const INLINE_RE = new RegExp(
  [
    "`([^`]+)`", // 1: `код`
    "\\*\\*([^*]+)\\*\\*", // 2: **жирный**
    "__([^_]+)__", // 3: __жирный__
    "\\*([^\\s*][^*]*?)\\*", // 4: *курсив*
    "\\[([^\\]]+)\\]\\(([^\\s)]+)\\)", // 5,6: [текст](ссылка)
  ].join("|"),
  "g",
);

/** Ссылки только http(s) и относительные (`/`, `#`): протоколы вроде
 * javascript: замалчиваются и остаются текстом. */
function safeHref(raw: string): string | null {
  if (/^https?:\/\//i.test(raw)) return raw;
  if (raw.startsWith("/") && !raw.startsWith("//")) return raw;
  if (raw.startsWith("#")) return raw;
  return null;
}

function renderInline(src: string, keyOffset = 0): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0;
  let key = keyOffset;
  let m: RegExpExecArray | null;
  INLINE_RE.lastIndex = 0;
  while ((m = INLINE_RE.exec(src)) !== null) {
    if (m.index > last) out.push(src.slice(last, m.index));
    const [full, code, bold, boldAlt, italic, linkText, linkUrl] = m;
    if (code !== undefined) {
      out.push(
        <code
          key={`i${key++}`}
          className="rounded bg-muted px-1 py-0.5 font-mono text-[0.85em] text-foreground"
        >
          {code}
        </code>,
      );
    } else if (bold !== undefined || boldAlt !== undefined) {
      out.push(<strong key={`i${key++}`}>{bold ?? boldAlt}</strong>);
    } else if (italic !== undefined) {
      out.push(<em key={`i${key++}`}>{italic}</em>);
    } else if (linkText !== undefined && linkUrl !== undefined) {
      const href = safeHref(linkUrl);
      if (href) {
        out.push(
          <a
            key={`i${key++}`}
            href={href}
            target={href.startsWith("http") ? "_blank" : undefined}
            rel={href.startsWith("http") ? "noopener noreferrer" : undefined}
            className="text-primary underline underline-offset-2"
          >
            {linkText}
          </a>,
        );
      } else {
        out.push(full);
      }
    }
    last = m.index + full.length;
  }
  if (last < src.length) out.push(src.slice(last));
  return out;
}

/* ------------------------------------------------------------------- публичное */

export function ChatMarkdown({
  children,
  className,
}: {
  children?: string | null;
  className?: string;
}): ReactNode {
  const src = children ?? "";
  if (!src.trim()) return null;
  const segments = parseSegments(src);

  return (
    <div className={cn("space-y-2 break-words", className)}>
      {segments.map((seg, i) => {
        switch (seg.kind) {
          case "code":
            return (
              <pre
                key={i}
                className="overflow-x-auto rounded-lg border border-border bg-muted/60 p-3 font-mono text-xs leading-relaxed text-foreground"
              >
                <code>{seg.code}</code>
              </pre>
            );
          case "heading":
            return (
              <h4 key={i} className="text-sm font-semibold text-foreground">
                {renderInline(seg.text)}
              </h4>
            );
          case "ul":
            return (
              <ul key={i} className="list-disc space-y-1 pl-5 text-sm">
                {seg.items.map((item, j) => (
                  <li key={j}>{renderInline(item)}</li>
                ))}
              </ul>
            );
          case "ol":
            return (
              <ol key={i} className="list-decimal space-y-1 pl-5 text-sm">
                {seg.items.map((item, j) => (
                  <li key={j}>{renderInline(item)}</li>
                ))}
              </ol>
            );
          case "hr":
            return <hr key={i} className="border-border" />;
          case "p":
          default:
            return (
              <p key={i} className="text-sm leading-relaxed">
                {renderInline(seg.text)}
              </p>
            );
        }
      })}
    </div>
  );
}
