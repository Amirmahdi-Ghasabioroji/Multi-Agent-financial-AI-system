import type { JsonRecord } from "@/lib/types";

export function titleCase(value: string) {
  return value
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

export function shortDate(value?: string) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

export function compactNumber(value: unknown) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return new Intl.NumberFormat("en", {
    notation: Math.abs(number) >= 10_000 ? "compact" : "standard",
    maximumFractionDigits: 2,
  }).format(number);
}

export function percent(value: unknown) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  const normalized = Math.abs(number) <= 1 ? number * 100 : number;
  return `${normalized.toFixed(normalized < 10 ? 1 : 0)}%`;
}

export function asRecord(value: unknown): JsonRecord {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonRecord)
    : {};
}

export function getValue(
  record: JsonRecord | undefined,
  keys: string[],
): unknown {
  if (!record) return undefined;
  for (const key of keys) {
    if (record[key] !== undefined && record[key] !== null) return record[key];
  }
  return undefined;
}

export function asStringList(value: unknown): string[] {
  if (!value) return [];
  if (Array.isArray(value)) {
    return value.map((item) =>
      typeof item === "string" ? item : JSON.stringify(item),
    );
  }
  return [String(value)];
}

export function parseJsonObject(value: string): {
  value?: JsonRecord;
  error?: string;
} {
  if (!value.trim()) return { value: {} };
  try {
    const parsed: unknown = JSON.parse(value);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return { error: "Enter a JSON object, for example { \"ticker\": \"AAPL\" }." };
    }
    return { value: parsed as JsonRecord };
  } catch (error) {
    return {
      error:
        error instanceof SyntaxError ? error.message : "The JSON is not valid.",
    };
  }
}

export function jobLabel(job: {
  query?: string;
  ticker?: string;
  tickers?: string[];
  kind?: string;
  agent?: string;
  payload?: JsonRecord;
}) {
  return (
    job.query?.trim() ||
    String(job.payload?.query ?? "").trim() ||
    job.ticker ||
    String(job.payload?.ticker ?? "").trim() ||
    job.tickers?.join(", ") ||
    titleCase(job.kind ?? job.agent ?? "Research run")
  );
}
