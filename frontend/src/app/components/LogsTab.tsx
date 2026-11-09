import { useState, useEffect, useCallback, useRef, useMemo, memo } from "react";
import { FileText, RefreshCw, AlertCircle, Search, Radio } from "lucide-react";
import { request } from "../../lib/api";
import {
  Badge,
  Button,
  Callout,
  Card,
  CardBody,
  CardHeader,
  EmptyState,
  PageHeader,
  inputClass,
} from "../../design";
import { cn } from "./ui/utils";

const MAX_LINES = 500;

// Токен в URL сокета больше не передаём: адрес попадает в логи прокси и
// историю браузера. Сессию браузер прикладывает cookie, сервер проверяет
// её через resolve_session до accept.
function wsUrl(): string {
  const proto = window.location.protocol === "https:" ? "wss://" : "ws://";
  return `${proto}${window.location.host}/api/admin/logs/ws`;
}

function LogsTabBase() {
  const [lines, setLines] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [level, setLevel] = useState<string>("all");
  const [search, setSearch] = useState("");
  const [live, setLive] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const [fallback, setFallback] = useState(false);
  const scrollRef = useRef<HTMLPreElement>(null);
  const linesRef = useRef<string[]>([]);

  const appendLines = useCallback((incoming: string[]) => {
    if (!incoming || incoming.length === 0) return;
    setLines((prev) => {
      const next = [...prev, ...incoming];
      return next.length > MAX_LINES ? next.slice(next.length - MAX_LINES) : next;
    });
  }, []);

  const loadLogs = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const q = level === "all" ? "?lines=400" : `?lines=400&level=${level}`;
      const data = await request<{ lines?: string[] }>(`/api/admin/logs/file${q}`);
      setLines(data.lines ?? []);
    } catch (e: any) {
      setError(e.message ?? "Ошибка сети");
    } finally {
      setLoading(false);
    }
  }, [level]);

  // Real-time streaming via WebSocket; fall back to 10s polling if it fails.
  useEffect(() => {
    setLoading(true);
    setError(null);
    let ws: WebSocket | null = null;
    let interval: number | null = null;
    let closed = false;

    try {
      ws = new WebSocket(wsUrl());
    } catch {
      ws = null;
    }

    if (ws) {
      ws.onopen = () => {
        if (!closed) {
          setLive(true);
          setFallback(false);
          setLoading(false);
        }
      };
      ws.onmessage = (ev) => {
        if (closed) return;
        try {
          const msg = JSON.parse(ev.data);
          if (Array.isArray(msg.lines)) {
            if (msg.type === "init") {
              linesRef.current = msg.lines.slice(-MAX_LINES);
              setLines(linesRef.current);
              setLoading(false);
            } else {
              appendLines(msg.lines);
            }
          }
        } catch {
          /* ignore malformed frames */
        }
      };
      ws.onerror = () => {
        /* will be handled by onclose */
      };
      ws.onclose = () => {
        if (closed) return;
        setLive(false);
        setFallback(true);
        setLoading(false);
        interval = window.setInterval(loadLogs, 10000);
      };
    } else {
      setFallback(true);
      setLoading(false);
      interval = window.setInterval(loadLogs, 10000);
    }

    return () => {
      closed = true;
      if (interval) window.clearInterval(interval);
      if (ws) {
        try {
          ws.close();
        } catch {
          /* ignore */
        }
      }
    };
  }, [loadLogs]);

  // Autoscroll to bottom when new lines arrive (only if enabled).
  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [lines, autoScroll]);

  const onScroll = () => {
    if (!scrollRef.current) return;
    const el = scrollRef.current;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    setAutoScroll(nearBottom);
  };

  // Чистый derive из state: фильтр не пересчитывается при каждом рендере
  // (WS-кадры, тик 10с-поллинга, ввод в поиске не меняют лишние зависимости).
  const filtered = useMemo(
    () =>
      search.trim()
        ? lines.filter((l) => l.toLowerCase().includes(search.toLowerCase()))
        : lines,
    [lines, search]
  );

  // Тяжёлый список (до 500 строк) - мемоизируем вывод: тик авто-скролла и
  // прочие ререндеры не пересоздают 500 DOM-узлов заново.
  const logRows = useMemo(
    () => filtered.map((line, i) => <div key={i} className={levelColor(line)}>{line}</div>),
    [filtered]
  );

  return (
    <div className="space-y-6">
      <PageHeader
        icon={FileText}
        title="Логи"
        description="Серверные логи в реальном времени с фильтром по уровню."
      />
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-muted-foreground">Системный лог бэкенда (logs/backend.log)</p>
        </div>
        <div className="flex items-center gap-3">
          <Badge tone={live ? "success" : "neutral"} className="gap-1">
            <Radio className="size-3" aria-hidden="true" />
            {live ? "Live" : fallback ? "Fallback (10с)" : "..."}
          </Badge>
          <Button variant="secondary" size="sm" onClick={loadLogs} disabled={loading}>
            <RefreshCw className={`size-4 ${loading ? "animate-spin" : ""}`} aria-hidden="true" />
            Обновить
          </Button>
        </div>
      </div>

      {error && (
        <Callout
          tone="danger"
          role="alert"
          icon={<AlertCircle className="size-5" aria-hidden="true" />}
        >
          {error}
        </Callout>
      )}

      <Card>
        <CardHeader
          title={
            <span className="flex items-center gap-2 text-lg">
              <FileText className="size-5" aria-hidden="true" />
              Хвост backend.log
              <Badge tone="neutral" className="text-xs">{lines.length} строк</Badge>
            </span>
          }
          description={
            <>
              Последние записи системного журнала бэкенда - обновляются в реальном времени
              {fallback ? " (WebSocket недоступен, автообновление раз в 10 секунд)" : ""}
            </>
          }
          actions={
            <div className="flex items-center gap-2">
              <div className="relative">
                <Search
                  className="size-4 absolute left-2.5 top-2.5 text-subtle-foreground"
                  aria-hidden="true"
                />
                <input
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Поиск по логам..."
                  aria-label="Поиск по логам"
                  className={cn(inputClass, "h-9 w-56 pl-9")}
                />
              </div>
              <select
                value={level}
                onChange={(e) => setLevel(e.target.value)}
                aria-label="Фильтр по уровню"
                className={cn(inputClass, "h-9 w-auto")}
              >
                <option value="all">Все уровни</option>
                <option value="error">Ошибки</option>
                <option value="warning">Предупреждения</option>
                <option value="info">Info</option>
                <option value="debug">Debug</option>
              </select>
            </div>
          }
        />
        <CardBody>
          {loading && <p className="text-sm text-muted-foreground">Загрузка...</p>}
          {!loading && filtered.length === 0 && (
            <EmptyState
              icon={<FileText className="size-8" aria-hidden="true" />}
              title="Записей не найдено"
              description="Измените фильтр по уровню или обновите список."
            />
          )}
          <pre
            ref={scrollRef}
            onScroll={onScroll}
            className="p-4 bg-muted border border-border rounded-lg overflow-x-auto text-xs leading-5 max-h-[70vh] overflow-y-auto font-mono whitespace-pre-wrap break-words"
          >
            {logRows}
          </pre>
        </CardBody>
      </Card>
    </div>
  );
}

// Чистая функция без доступа к стейту - вынесена на уровень модуля,
// чтобы useMemo по filtered не зависел от рекреации на каждом рендере.
// Тональности - дизайн-токены: ошибка - destructive, предупреждение -
// warning, debug - muted-foreground, обычные строки - success.
function levelColor(line: string) {
  if (line.includes("level=error") || line.startsWith("ERROR ")) return "text-destructive";
  if (line.includes("level=warning") || line.startsWith("WARNING ") || line.startsWith("WARN ")) return "text-warning";
  if (line.includes("level=debug")) return "text-muted-foreground";
  return "text-success";
}

export const LogsTab = memo(LogsTabBase);
