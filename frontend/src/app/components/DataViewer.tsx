import { Database } from "lucide-react";
import {
  Badge,
  Card,
  CardBody,
  CardHeader,
  TableFrame,
  tableHeadClass,
  tableRowClass,
} from "../../design";

function ValueDisplay({ value }: { value: unknown }) {
  if (value === null || value === undefined) {
    return <span className="text-subtle-foreground italic">-</span>;
  }
  if (typeof value === "boolean") {
    return <Badge tone={value ? "primary" : "neutral"}>{String(value)}</Badge>;
  }
  if (typeof value === "number") {
    return <span className="font-mono text-primary">{value}</span>;
  }
  if (typeof value === "string") {
    if (value.length > 120) return <span className="text-xs">{value.slice(0, 120)}…</span>;
    return <span>{value}</span>;
  }
  if (Array.isArray(value)) {
    return <ArrayViewer data={value} />;
  }
  if (typeof value === "object") {
    return <ObjectViewer data={value as Record<string, unknown>} depth={0} />;
  }
  return <span>{String(value)}</span>;
}

function ArrayViewer({ data }: { data: unknown[] }) {
  if (data.length === 0) return <span className="text-subtle-foreground italic">пусто</span>;
  const allPrimitives = data.every((v) => typeof v !== "object" || v === null);
  if (allPrimitives) {
    return (
      <div className="flex flex-wrap gap-1">
        {data.map((v, i) => (
          <Badge key={i} tone="neutral" className="text-xs">
            {String(v ?? "")}
          </Badge>
        ))}
      </div>
    );
  }
  const allObjects = data.every((v) => typeof v === "object" && v !== null);
  if (allObjects && data.length > 0) {
    const keys = new Set<string>();
    data.forEach((obj) => Object.keys(obj as object).forEach((k) => keys.add(k)));
    const keyArr = Array.from(keys).slice(0, 8);
    return (
      <TableFrame>
        <table className="w-full text-sm">
          <thead className={tableHeadClass}>
            <tr>
              <th scope="col" className="px-3 py-2 text-left font-semibold">#</th>
              {keyArr.map((k) => (
                <th
                  key={k}
                  scope="col"
                  className="px-3 py-2 text-left font-semibold whitespace-nowrap"
                >
                  {k}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.slice(0, 50).map((obj, i) => (
              <tr key={i} className={tableRowClass}>
                <td className="px-3 py-2 text-xs text-muted-foreground">{i + 1}</td>
                {keyArr.map((k) => (
                  <td key={k} className="px-3 py-2 text-xs max-w-[200px] truncate">
                    <ValueDisplay value={(obj as Record<string, unknown>)[k]} />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {data.length > 50 && (
          <div className="px-3 py-2 text-xs text-muted-foreground border-t border-border bg-muted">
            и ещё {data.length - 50} записей
          </div>
        )}
      </TableFrame>
    );
  }
  return <span className="text-xs text-muted-foreground">[{data.length} элементов]</span>;
}

function ObjectViewer({
  data,
  depth,
}: {
  data: Record<string, unknown>;
  depth: number;
}) {
  const entries = Object.entries(data).filter(
    ([, v]) => v !== null && v !== undefined
  );
  if (entries.length === 0) return <span className="text-subtle-foreground italic">пусто</span>;
  if (depth > 2) {
    return <span className="text-xs text-muted-foreground">{entries.length} полей</span>;
  }
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
      {entries.map(([k, v]) => (
        <div key={k} className="border border-border rounded-md p-3 bg-card">
          <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-1">
            {k.replace(/_/g, " ")}
          </div>
          <ValueDisplay value={v} />
        </div>
      ))}
    </div>
  );
}

export function DataViewer({ data }: { data: unknown }) {

  if (!data) return null;

  return (
    <div className="space-y-4">
      {/* Structured view */}
      <Card>
        <CardHeader
          title={
            <span className="flex items-center gap-2">
              <Database className="size-4 text-primary" aria-hidden="true" />
              Структурированные данные
            </span>
          }
        />
        <CardBody>
          <ValueDisplay value={data} />
        </CardBody>
      </Card>
    </div>
  );
}
