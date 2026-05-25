import { Database, ChevronDown, ChevronRight, FileJson } from "lucide-react";
import { useState } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./ui/table";
import { Badge } from "./ui/badge";

function ValueDisplay({ value }: { value: unknown }) {
  if (value === null || value === undefined) {
    return <span className="text-gray-400 italic">—</span>;
  }
  if (typeof value === "boolean") {
    return <Badge variant={value ? "default" : "secondary"}>{String(value)}</Badge>;
  }
  if (typeof value === "number") {
    return <span className="font-mono text-blue-700">{value}</span>;
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
  if (data.length === 0) return <span className="text-gray-400 italic">пусто</span>;
  const allPrimitives = data.every((v) => typeof v !== "object" || v === null);
  if (allPrimitives) {
    return (
      <div className="flex flex-wrap gap-1">
        {data.map((v, i) => (
          <Badge key={i} variant="outline" className="text-xs">
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
      <div className="border rounded-md overflow-hidden">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="text-xs font-medium">#</TableHead>
              {keyArr.map((k) => (
                <TableHead key={k} className="text-xs font-medium whitespace-nowrap">
                  {k}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {data.slice(0, 50).map((obj, i) => (
              <TableRow key={i}>
                <TableCell className="text-xs text-gray-500">{i + 1}</TableCell>
                {keyArr.map((k) => (
                  <TableCell key={k} className="text-xs max-w-[200px] truncate">
                    <ValueDisplay value={(obj as Record<string, unknown>)[k]} />
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
        {data.length > 50 && (
          <div className="px-3 py-2 text-xs text-gray-500 border-t bg-gray-50">
            и ещё {data.length - 50} записей
          </div>
        )}
      </div>
    );
  }
  return <span className="text-xs text-gray-500">[{data.length} элементов]</span>;
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
  if (entries.length === 0) return <span className="text-gray-400 italic">пусто</span>;
  if (depth > 2) {
    return <span className="text-xs text-gray-500">{entries.length} полей</span>;
  }
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
      {entries.map(([k, v]) => (
        <div key={k} className="border rounded-md p-3 bg-white">
          <div className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">
            {k.replace(/_/g, " ")}
          </div>
          <ValueDisplay value={v} />
        </div>
      ))}
    </div>
  );
}

export function DataViewer({ data }: { data: unknown }) {
  const [showRaw, setShowRaw] = useState(false);

  if (!data) return null;

  return (
    <div className="space-y-4">
      {/* Structured view */}
      <div className="border border-gray-200 rounded-lg p-4 bg-white">
        <div className="flex items-center gap-2 mb-3">
          <Database className="size-4 text-emerald-600" />
          <span className="text-sm font-medium text-gray-900">Структурированные данные</span>
        </div>
        <ValueDisplay value={data} />
      </div>

      {/* Raw JSON toggle */}
      <details className="border border-gray-200 rounded-lg">
        <summary
          className="px-4 py-2 text-sm font-medium text-gray-500 cursor-pointer hover:bg-gray-50 rounded-lg select-none flex items-center gap-2"
          onClick={(e) => {
            e.preventDefault();
            setShowRaw(!showRaw);
          }}
        >
          {showRaw ? (
            <ChevronDown className="size-3.5" />
          ) : (
            <ChevronRight className="size-3.5" />
          )}
          <FileJson className="size-3.5" />
          JSON
        </summary>
        {showRaw && (
          <pre className="p-4 text-xs text-gray-600 overflow-auto max-h-96 whitespace-pre-wrap font-mono bg-gray-50 rounded-b-lg border-t">
            {JSON.stringify(data, null, 2)}
          </pre>
        )}
      </details>
    </div>
  );
}
