import { useState } from "react";
import { ChevronRight, ChevronDown } from "lucide-react";

interface CompetencyNode {
  code: string;
  total_skills?: number;
  matched_skills?: number;
  coverage?: number;
  weighted_coverage?: number;
  children?: CompetencyNode[];
}

function buildTree(flat: CompetencyNode[]): CompetencyNode[] {
  const nodeMap = new Map<string, CompetencyNode & { _children: CompetencyNode[] }>();
  const roots: CompetencyNode[] = [];

  for (const node of flat) {
    nodeMap.set(node.code, { ...node, _children: [] });
  }

  for (const node of flat) {
    const dotIdx = node.code.lastIndexOf(".");
    if (dotIdx > 0) {
      const parentCode = node.code.slice(0, dotIdx);
      const parent = nodeMap.get(parentCode);
      if (parent) {
        parent._children.push(nodeMap.get(node.code)!);
        continue;
      }
    }
    roots.push(nodeMap.get(node.code)!);
  }

  for (const node of nodeMap.values()) {
    (node as any).children = node._children;
    delete (node as any)._children;
  }

  return roots;
}

function coverageColor(val: number | undefined): string {
  if (val === undefined) return "text-gray-300";
  if (val >= 0.6) return "text-green-600";
  if (val >= 0.2) return "text-yellow-600";
  return "text-red-600";
}

function bgColor(val: number | undefined): string {
  if (val === undefined) return "bg-gray-50";
  if (val >= 0.6) return "bg-green-50 border-green-200";
  if (val >= 0.2) return "bg-yellow-50 border-yellow-200";
  return "bg-red-50 border-red-200";
}

function TreeNode({ node, depth }: { node: CompetencyNode; depth: number }) {
  const [open, setOpen] = useState(depth < 2);
  const hasChildren = node.children && node.children.length > 0;
  const childCov = hasChildren
    ? node.children!.reduce((s, c) => s + (c.weighted_coverage ?? c.coverage ?? 0), 0) / node.children!.length
    : undefined;
  const displayCov = node.weighted_coverage ?? (node.total_skills === 0 ? childCov : node.coverage) ?? childCov;
  const matched = node.matched_skills ?? 0;
  const total = node.total_skills ?? 0;

  return (
    <div>
      <div
        className={`flex items-center justify-between px-3 py-2 text-sm border rounded mb-0.5 cursor-pointer transition-colors hover:bg-gray-100 ${bgColor(displayCov)}`}
        style={{ marginLeft: depth * 16 }}
        onClick={() => hasChildren && setOpen(!open)}
      >
        <div className="flex items-center gap-2 min-w-0">
          {hasChildren ? (
            open ? <ChevronDown className="size-4 shrink-0 text-gray-400" /> : <ChevronRight className="size-4 shrink-0 text-gray-400" />
          ) : (
            <span className="size-4 inline-block" />
          )}
          <span className="font-medium text-gray-800 truncate">{node.code}</span>
          {hasChildren && (
            <span className="text-xs text-gray-400 shrink-0">({node.children!.length})</span>
          )}
        </div>
        <div className="flex items-center gap-3 shrink-0 ml-3">
          <span className={`text-xs ${total > 0 ? "text-gray-500" : "text-gray-300"}`}>
            {matched}/{total}
          </span>
          <span className={`font-semibold text-sm tabular-nums ${coverageColor(displayCov)}`}>
            {displayCov !== undefined ? (displayCov * 100).toFixed(0) : "—"}%
          </span>
        </div>
      </div>
      {open && hasChildren && (
        <div>
          {node.children!.map((child) => (
            <TreeNode key={child.code} node={child} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  );
}

export function CompetencyTree({
  competencies,
  className = "",
}: {
  competencies: CompetencyNode[];
  className?: string;
}) {
  const tree = buildTree(competencies);

  if (tree.length === 0) return null;

  return (
    <div className={`divide-y divide-gray-100 border rounded-lg ${className}`}>
      {tree.map((root) => (
        <TreeNode key={root.code} node={root} depth={0} />
      ))}
    </div>
  );
}
