import { useState } from "react";
import { ChevronRight, ChevronDown } from "lucide-react";

import { Badge, type BadgeTone } from "../../design";

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

function coverageTone(val: number | undefined): BadgeTone {
  if (val === undefined) return "neutral";
  if (val >= 0.6) return "success";
  if (val >= 0.2) return "warning";
  return "danger";
}

function rowColor(val: number | undefined): string {
  if (val === undefined) return "bg-muted";
  if (val >= 0.6) return "bg-success-soft border-success/30";
  if (val >= 0.2) return "bg-warning-soft border-warning/30";
  return "bg-destructive/5 border-destructive/30";
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

/**
 * Строка узла дерева. Отступ уровней - вложенные `pl-4` обёртки
 * (16px на уровень), а не inline-стили: геометрия та же, что была при
 * `marginLeft: depth * 16` (список навыков: 16px * depth + 20px + px-3
 * = `pl-8`). Строка - кнопка: дерево работает и с клавиатуры.
 * Все цвета - дизайн-токены, светлая и тёмная темы без правок.
 */
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
  const expandable = Boolean(hasChildren || hasSkills);

  return (
    <div>
      <button
        type="button"
        aria-expanded={expandable ? open : undefined}
        onClick={() => (hasChildren || hasSkills) && setOpen(!open)}
        className={`flex w-full items-center justify-between px-3 py-2 text-left text-sm rounded border mb-0.5 cursor-pointer transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background ${rowColor(displayCov)}`}
      >
        <span className="flex items-center gap-2 min-w-0">
          {(hasChildren && open) ? (
            <ChevronDown className="size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
          ) : hasChildren ? (
            <ChevronRight className="size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
          ) : (
            <span className="size-4 inline-block" aria-hidden="true" />
          )}
          <span className="font-medium text-foreground truncate">{node.code}</span>
          {hasChildren && (
            <Badge tone="neutral" className="shrink-0">
              ({node.children!.length})
            </Badge>
          )}
        </span>
        <span className="flex items-center gap-3 shrink-0 ml-3">
          <Badge tone="neutral" className="text-xs tabular-nums">
            {matched}/{total}
          </Badge>
          <Badge tone={coverageTone(displayCov)} className="font-semibold text-sm tabular-nums">
            {displayCov !== undefined ? (displayCov * 100).toFixed(0) : "-"}%
          </Badge>
        </span>
      </button>
      {open && hasChildren && (
        <div className="pl-4">
          {node.children!.map((child) => (
            <TreeNode key={child.code} node={child} depth={depth + 1} />
          ))}
        </div>
      )}
      {open && hasSkills && (
        <div className="flex flex-wrap gap-1 py-2 pl-8 pr-3">
          {matchedList.slice(0, 8).map((s, i) => (
            <Badge key={"m" + i} tone="success" title={s}>
              {shortSkill(s)}
            </Badge>
          ))}
          {gapList.slice(0, 8).map((s, i) => (
            <Badge key={"g" + i} tone="danger" title={s}>
              {shortSkill(s)}
            </Badge>
          ))}
          {(matchedList.length > 8 || gapList.length > 8) && (
            <Badge tone="neutral" className="self-center">
              +{Math.max(0, matchedList.length - 8) + Math.max(0, gapList.length - 8)} more
            </Badge>
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
    <div className={`divide-y divide-border border rounded-lg bg-card ${className}`}>
      {tree.map((root) => (
        <TreeNode key={root.code} node={root} depth={0} />
      ))}
    </div>
  );
}
