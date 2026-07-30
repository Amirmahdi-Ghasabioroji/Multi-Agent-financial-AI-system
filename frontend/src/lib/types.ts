export type AgentKind = "analyst" | "risk" | "strategy" | "execution";
export type RunKind = AgentKind | "pipeline" | "evaluation";
export type JobStatus =
  | "queued"
  | "pending"
  | "running"
  | "succeeded"
  | "completed"
  | "failed"
  | "cancelled";

export type JsonRecord = Record<string, unknown>;

export interface Citation {
  id?: string;
  title?: string;
  source?: string;
  url?: string;
  date?: string;
  score?: number;
  excerpt?: string;
}

export interface Conversation {
  id: string;
  title?: string;
  created_at?: string;
  updated_at?: string;
  message_count?: number;
}

export interface StageState {
  name: string;
  status: JobStatus | "waiting";
  message?: string;
  started_at?: string;
  completed_at?: string;
}

export interface Job {
  id: string;
  status: JobStatus;
  kind?: RunKind;
  agent?: RunKind;
  query?: string;
  ticker?: string;
  tickers?: string[];
  mode?: "demo" | "live";
  demo?: boolean;
  metadata?: JsonRecord;
  payload?: JsonRecord;
  conversation_id?: string;
  created_at?: string;
  updated_at?: string;
  started_at?: string;
  completed_at?: string;
  progress?: number;
  stage?: string;
  stages?: StageState[];
  result?: MAFASResult;
  partial_result?: MAFASResult;
  error?: string | JsonRecord;
}

export interface Health {
  status?: string;
  service?: string;
  version?: string;
  mode?: string;
  llm_available?: boolean;
  vector_store?: string;
  collection_count?: number;
  document_count?: number;
  corpus_updated_at?: string;
  providers?: JsonRecord;
  services?: Record<string, JsonRecord>;
  [key: string]: unknown;
}

export interface AppConfig {
  demo_mode?: boolean;
  llm_enabled?: boolean;
  collection_name?: string;
  data_mode?: string;
  watchlist?: string[];
  playbooks?: JsonRecord[];
  job_kinds?: string[];
  [key: string]: unknown;
}

export interface RunRequest {
  query?: string;
  ticker?: string;
  tickers?: string[];
  lookback_days?: number;
  use_llm?: boolean;
  demo?: boolean;
  demo_mode?: boolean;
  conversation_id?: string;
  context?: string | JsonRecord | JsonRecord[];
  briefing?: JsonRecord;
  setups?: JsonRecord[];
  risk?: JsonRecord;
  [key: string]: unknown;
}

export interface MAFASResult {
  analyst?: JsonRecord;
  briefing?: JsonRecord;
  risk?: JsonRecord;
  strategy?: JsonRecord;
  execution?: JsonRecord;
  cards?: JsonRecord[];
  aggregate?: JsonRecord;
  decision?: string;
  no_trade_reason?: string;
  route_log?: string[];
  no_trade?: boolean;
  route?: string | string[];
  errors?: unknown[];
  citations?: Citation[];
  [key: string]: unknown;
}

export interface JobEvent {
  type?: string;
  event_type?: string;
  event?: string;
  status?: JobStatus;
  stage?: string;
  message?: string;
  progress?: number;
  result?: MAFASResult;
  partial_result?: MAFASResult;
  data?: unknown;
  [key: string]: unknown;
}

export interface ListResponse<T> {
  items: T[];
  total?: number;
}
