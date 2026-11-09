import { useState } from "react";
import { ChevronRight, ChevronDown } from "lucide-react";

interface CompetencyNode {
  code: string;
  total_skills?: number;
  matched_skills?: number;
  coverage?: number;
  weighted_coverage?: number;
  matched?: string[];
  gaps?: string[];
  children?: CompetencyNode[];
}

function shortSkill(s: string): string {
  return s.length > 35 ? s.slice(0, 35) + "\u2026" : s;
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
  if (val === undefined) return "text-subtle-foreground";
  if (val >= 0.6) return "text-success";
  if (val >= 0.2) return "text-warning";
  return "text-destructive";
}

function bgColor(val: number | undefined): string {
  if (val === undefined) return "bg-muted";
  if (val >= 0.6) return "bg-success-soft border-success/30";
  if (val >= 0.2) return "bg-warning-soft border-warning/30";
  return "bg-destructive/10 border-destructive/30";
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
  const matchedList = node.matched ?? [];
  const gapList = node.gaps ?? [];
  const hasSkills = matchedList.length + gapList.length > 0;

  return (
    <div>
      <div
        className={`flex items-center justify-between px-3 py-2 text-sm border rounded mb-0.5 cursor-pointer transition-colors hover:bg-muted ${bgColor(displayCov)}`}
        style={{ marginLeft: depth * 16 }}
        onClick={() => (hasChildren || hasSkills) && setOpen(!open)}
      >
        <div className="flex items-center gap-2 min-w-0">
          {hasChildren ? (
            open ? <ChevronDown className="size-4 shrink-0 text-subtle-foreground" /> : <ChevronRight className="size-4 shrink-0 text-subtle-foreground" />
          ) : (
            <span className="size-4 inline-block" />
          )}
          <span className="font-medium text-foreground truncate">{node.code}</span>
          {hasChildren && (
            <span className="text-xs text-subtle-foreground shrink-0">({node.children!.length})</span>
          )}
        </div>
        <div className="flex items-center gap-3 shrink-0 ml-3">
          <span className={`text-xs ${total > 0 ? "text-muted-foreground" : "text-subtle-foreground"}`}>
            {matched}/{total}
          </span>
          <span className={`font-semibold text-sm tabular-nums ${coverageColor(displayCov)}`}>
            {displayCov !== undefined ? (displayCov * 100).toFixed(0) : "-"}%
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
      {open && hasSkills && (
        <div className="flex flex-wrap gap-1 px-3 py-2" style={{ marginLeft: depth * 16 + 20 }}>
          {matchedList.slice(0, 8).map((s, i) => (
            <span key={"m" + i} title={s} className="text-xs px-2 py-0.5 rounded-full bg-success-soft text-success border border-success/30">
              {shortSkill(s)}
            </span>
          ))}
          {gapList.slice(0, 8).map((s, i) => (
            <span key={"g" + i} title={s} className="text-xs px-2 py-0.5 rounded-full bg-destructive/10 text-destructive border border-destructive/30">
              {shortSkill(s)}
            </span>
          ))}
          {(matchedList.length > 8 || gapList.length > 8) && (
            <span className="text-xs px-2 py-0.5 text-subtle-foreground">
              +{Math.max(0, matchedList.length - 8) + Math.max(0, gapList.length - 8)} more
            </span>
          )}
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
    <div className={`divide-y divide-border border rounded-lg ${className}`}>
      {tree.map((root) => (
        <TreeNode key={root.code} node={root} depth={0} />
      ))}
    </div>
  );
}
