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
        className="inline-flex items-center gap-1.5 rounded-full border border-border-strong bg-card px-4 py-1.5 text-xs font-medium text-muted-foreground shadow-sm transition-all hover:border-primary hover:text-primary hover:shadow cursor-pointer"
      >
        <ChevronDown
          className={cn("size-3.5 transition-transform duration-200", expanded && "rotate-180")}
        />
        {expanded ? "Свернуть" : `Показать ещё ${total - shown} из ${total}`}
      </button>
    </div>
  );
}
