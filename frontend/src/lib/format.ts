import type { JsonRecord } from "@/lib/types";

export function titleCase(value: string) {
  return value
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function parseTradeDate(value: string): Date | null {
  const match = /^(\d{4})-(\d{2})-(\d{2})/.exec(value);
  if (match) {
    const [, year, month, day] = match;
    return new Date(Number(year), Number(month) - 1, Number(day));
  }
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

export function tradeDatePrice(dateValue?: string, priceValue?: unknown) {
  const price = Number(priceValue);
  const hasPrice = Number.isFinite(price);
  if (!dateValue && !hasPrice) return "—";

  let datePart = "—";
  if (dateValue) {
    const date = parseTradeDate(dateValue);
    if (date) {
      datePart = new Intl.DateTimeFormat("en-GB", {
        day: "2-digit",
        month: "short",
        year: "numeric",
      }).format(date);
    } else {
      datePart = dateValue;
    }
  }

  if (!hasPrice) return datePart;

  const pricePart = new Intl.NumberFormat("en", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
    useGrouping: false,
  }).format(price);

  return `${datePart} @ ${pricePart}`;
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

function readString(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function readTickerList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => (typeof item === "string" ? item.trim().toUpperCase() : ""))
    .filter(Boolean);
}

export function jobQuery(job: {
  query?: string;
  payload?: JsonRecord;
  result?: JsonRecord;
}): string {
  const payload = asRecord(job.payload);
  const result = asRecord(job.result);
  const briefing = asRecord(result.briefing);
  return (
    readString(job.query) ||
    readString(payload.query) ||
    readString(result.query) ||
    readString(briefing.query) ||
    readString(result.analyst_query) ||
    readString(payload.analyst_query)
  );
}

export function jobTickers(job: {
  ticker?: string;
  tickers?: string[];
  payload?: JsonRecord;
  result?: JsonRecord;
}): string[] {
  const payload = asRecord(job.payload);
  const result = asRecord(job.result);
  const risk = asRecord(result.risk);
  const strategy = asRecord(result.strategy);

  for (const list of [
    job.tickers,
    payload.tickers,
    result.tickers,
    risk.universe,
    strategy.universe,
  ]) {
    const tickers = readTickerList(list);
    if (tickers.length) return tickers;
  }

  for (const value of [job.ticker, payload.ticker, result.ticker]) {
    const ticker = readString(value).toUpperCase();
    if (ticker) return [ticker];
  }

  return [];
}

export function jobLabel(job: {
  query?: string;
  ticker?: string;
  tickers?: string[];
  kind?: string;
  agent?: string;
  payload?: JsonRecord;
  result?: JsonRecord;
}) {
  const query = jobQuery(job);
  if (query) return query;
  const tickers = jobTickers(job);
  if (tickers.length) return tickers.join(", ");
  return titleCase(job.kind ?? job.agent ?? "Research run");
}
