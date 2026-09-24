import { ChevronDown } from "lucide-react";
import { cn } from "./utils";

interface ShowMoreProps {
  total: number;
  shown: number;
  expanded: boolean;
  onToggle: () => void;
  className?: string;
}

/** Компактная пилюля «показать ещё / свернуть» для progressive disclosure списков. */
export function ShowMore({ total, shown, expanded, onToggle, className }: ShowMoreProps) {
  if (total <= shown) return null;
  return (
    <div className={cn("flex justify-center pt-1", className)}>
      <button
        onClick={onToggle}
        aria-expanded={expanded}
        className="inline-flex items-center gap-1.5 rounded-full border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-950 px-4 py-1.5 text-xs font-medium text-slate-600 dark:text-slate-300 shadow-sm transition-all hover:border-blue-400 dark:hover:border-blue-500 hover:text-blue-700 dark:hover:text-blue-300 hover:shadow cursor-pointer"
      >
        <ChevronDown
          className={cn("size-3.5 transition-transform duration-200", expanded && "rotate-180")}
        />
        {expanded ? "Свернуть" : `Показать ещё ${total - shown} из ${total}`}
      </button>
    </div>
  );
}
