"use client";

import * as React from "react";

function Progress({
  className,
  value,
  ...props
}: { className?: string; value?: number } & React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      data-slot="progress"
      className={"relative h-2 w-full overflow-hidden rounded-full bg-primary/20 " + (className || "")}
      {...props}
    >
      <div
        data-slot="progress-indicator"
        className="h-full bg-primary transition-all"
        style={{ width: `${Math.min(100, Math.max(0, value || 0))}%` }}
      />
    </div>
  );
}

export { Progress };
