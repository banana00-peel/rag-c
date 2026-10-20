export type DocumentStatus = "pending" | "processing" | "completed" | "failed";

export interface DocumentRecord {
  id: string;
  filename: string;
  brand: string;
  model: string;
  status: DocumentStatus;
  page_count: number;
  chunk_count: number;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface SourceRecord {
  source_id: string;
  document_id: string;
  document_name: string;
  page_start: number;
  page_end: number;
  model: string;
  snippet: string;
  chunk_ids: string[];
  dense_score: number | null;
  dense_distance: number | null;
  bm25_score: number | null;
  rrf_score: number;
}

export interface ChatResponse {
  answer: string;
  refused: boolean;
  refusal_code:
    | "no_recall"
    | "low_relevance"
    | "insufficient_evidence"
    | "invalid_citation"
    | null;
  refusal_reason: string | null;
  sources: SourceRecord[];
  session_id: string;
  standalone_query: string;
  history_used: boolean;
}

export interface UploadResponse {
  document: DocumentRecord;
  duplicate: boolean;
}

