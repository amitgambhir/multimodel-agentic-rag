export type Modality = "text" | "url" | "pdf" | "image";

export interface Source {
  id: string;
  title: string;
  modality: Modality;
  chunk_count: number;
  created_at: number;
  notes?: string;
  preview?: string;
}

export interface Match {
  source_id: string;
  source: string;
  modality: Modality;
  chunk_index: number;
  snippet: string;
  score: number;
}

export interface Point {
  source_id: string;
  title: string;
  modality: Modality;
  preview?: string;
  x: number;
  y: number;
  z: number;
}

export interface Snapshot {
  sources: Source[];
  points: Point[];
  query_point?: { x: number; y: number; z: number } | null;
  events: Array<{ kind: string; source_id: string; title: string; modality: Modality; at: number }>;
  dimensions: number;
  embedding_provider: string;
}

export interface Usage {
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cache_creation_tokens: number;
}

export interface TraceStep { step: string; detail: string }

export type StreamEvent =
  | { type: "trace"; step: string; detail: string }
  | { type: "answer_delta"; delta: string }
  | { type: "reset_answer" }
  | { type: "tool"; name: string; input: any; result_summary: string }
  | { type: "usage"; input_tokens: number; output_tokens: number; cache_read_tokens: number; cache_creation_tokens: number }
  | { type: "done"; answer: string; matches: Match[]; trace: TraceStep[]; usage: Usage; space: Snapshot }
  | { type: "error"; message: string };

export type Provider = "claude" | "gemini";
