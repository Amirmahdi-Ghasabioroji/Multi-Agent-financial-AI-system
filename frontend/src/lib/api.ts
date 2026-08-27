import type {
  AppConfig,
  Conversation,
  Health,
  Job,
  JobEvent,
  ListResponse,
  RunKind,
  RunRequest,
} from "@/lib/types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ??
  "http://localhost:8000/api/v1";

export class ApiError extends Error {
  constructor(
    message: string,
    public status?: number,
    public detail?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function unwrap<T>(value: unknown): T {
  if (
    value &&
    typeof value === "object" &&
    "data" in value &&
    Object.keys(value).length <= 3
  ) {
    return (value as { data: T }).data;
  }
  return value as T;
}

function asList<T>(value: unknown): T[] {
  const unwrapped = unwrap<unknown>(value);
  if (Array.isArray(unwrapped)) return unwrapped as T[];
  if (unwrapped && typeof unwrapped === "object") {
    const record = unwrapped as Record<string, unknown>;
    for (const key of ["items", "jobs", "conversations", "results"]) {
      if (Array.isArray(record[key])) return record[key] as T[];
    }
  }
  return [];
}

async function request<T>(
  paths: string | string[],
  init?: RequestInit,
): Promise<T> {
  const candidates = Array.isArray(paths) ? paths : [paths];
  let lastError: ApiError | undefined;

  for (const path of candidates) {
    try {
      const response = await fetch(`${API_BASE}${path}`, {
        ...init,
        headers: {
          Accept: "application/json",
          ...(init?.body ? { "Content-Type": "application/json" } : {}),
          ...init?.headers,
        },
        cache: "no-store",
      });

      if (!response.ok) {
        let detail: unknown;
        try {
          detail = await response.json();
        } catch {
          detail = await response.text();
        }
        const message =
          typeof detail === "object" &&
          detail &&
          "detail" in detail &&
          typeof detail.detail === "string"
            ? detail.detail
            : `Request failed (${response.status})`;
        lastError = new ApiError(message, response.status, detail);
        if (response.status === 404 || response.status === 405) continue;
        throw lastError;
      }

      if (response.status === 204) return undefined as T;
      return unwrap<T>(await response.json());
    } catch (error) {
      if (error instanceof ApiError) {
        lastError = error;
        if (error.status !== 404 && error.status !== 405) throw error;
      } else {
        lastError = new ApiError(
          error instanceof Error ? error.message : "Could not reach the API",
        );
      }
    }
  }

  throw lastError ?? new ApiError("No compatible API route was found");
}

export const api = {
  health: () => request<Health>(["/health", "/system/health"]),
  config: () =>
    request<AppConfig>(["/config/defaults", "/config", "/system/config"]),

  async jobs(): Promise<ListResponse<Job>> {
    const data = await request<unknown>(["/jobs", "/runs"]);
    return {
      items: asList<Job>(data),
      total:
        data && typeof data === "object" && "total" in data
          ? Number((data as { total: unknown }).total)
          : undefined,
    };
  },

  job: (id: string) =>
    request<Job>([`/jobs/${encodeURIComponent(id)}`, `/runs/${encodeURIComponent(id)}`]),

  async conversations(): Promise<ListResponse<Conversation>> {
    const data = await request<unknown>(["/conversations", "/threads"]);
    return { items: asList<Conversation>(data) };
  },

  createConversation: (title: string) =>
    request<Conversation>("/conversations", {
      method: "POST",
      body: JSON.stringify({ title }),
    }),

  run: (kind: RunKind, payload: RunRequest) =>
    request<Job>(
      [`/runs/${kind}`, `/${kind}/run`, `/agents/${kind}/run`],
      { method: "POST", body: JSON.stringify(payload) },
    ),

  runEvaluation: (payload: {
    suites?: Array<"rag" | "simulation" | "risk" | "analyst" | "gates" | "strategy" | "all">;
    top_k?: number;
    lookback_days?: number;
    tickers?: string[];
  }) =>
    request<Job>(["/evaluation/run", "/runs/evaluation"], {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  deleteJob: (id: string) =>
    request<void>(
      [`/jobs/${encodeURIComponent(id)}`, `/runs/${encodeURIComponent(id)}`],
      { method: "DELETE" },
    ),

  refreshCorpus: () =>
    request<unknown>(["/corpus/refresh", "/data/refresh"], {
      method: "POST",
      body: JSON.stringify({}),
    }),

  resetCorpus: () =>
    request<unknown>(["/corpus/reset", "/data/reset"], {
      method: "POST",
      body: JSON.stringify({ confirmation: "RESET FINANCIAL DOCS" }),
    }),

  exportUrl: (id: string, format: "json" | "markdown") =>
    `${API_BASE}/jobs/${encodeURIComponent(id)}/export/${format}`,

  eventUrl: (id: string) =>
    `${API_BASE}/jobs/${encodeURIComponent(id)}/events`,
};

export function subscribeToJob(
  id: string,
  handlers: {
    onEvent: (event: JobEvent) => void;
    onError: () => void;
  },
) {
  const source = new EventSource(api.eventUrl(id));
  const read = (event: MessageEvent<string>) => {
    try {
      const parsed = JSON.parse(event.data) as JobEvent;
      handlers.onEvent({
        ...parsed,
        type: parsed.event_type ?? parsed.type ?? event.type,
      });
    } catch {
      handlers.onEvent({ type: event.type, message: event.data });
    }
  };

  source.onmessage = read;
  for (const eventName of [
    "status",
    "stage",
    "progress",
    "partial",
    "result",
    "complete",
    "queued",
    "running",
    "succeeded",
    "failed",
    "stage_started",
    "stage_completed",
    "stage_failed",
    "stage_skipped",
    "pipeline_started",
    "pipeline_completed",
    "no_trade",
    "corpus_completed",
    "error",
  ]) {
    source.addEventListener(eventName, read as EventListener);
  }
  source.onerror = handlers.onError;
  return () => source.close();
}

export function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : "An unexpected error occurred.";
}
