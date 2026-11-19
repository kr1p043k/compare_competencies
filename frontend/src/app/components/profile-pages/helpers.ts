export const CUSTOM_NAME_RE = /^[a-z0-9_-]{2,40}$/;

export const NO_BASE = "__none__";

export const splitLines = (s: string): string[] =>
  s.split("\n").map((l) => l.trim()).filter((l) => l.length > 0);

export const norm = (s: string) => s.toLowerCase().trim();
