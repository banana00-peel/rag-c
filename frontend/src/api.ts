import type {
  ChatResponse,
  DocumentRecord,
  UploadResponse,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...init?.headers,
    },
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail ?? `请求失败：${response.status}`);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return response.json() as Promise<T>;
}

export function listDocuments(): Promise<DocumentRecord[]> {
  return request<DocumentRecord[]>("/documents");
}

export function uploadDocument(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  return request<UploadResponse>("/documents/upload", {
    method: "POST",
    body: formData,
  });
}

export function updateDocument(
  id: string,
  payload: { brand: string; model: string },
): Promise<DocumentRecord> {
  return request<DocumentRecord>(`/documents/${id}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

export function startIndexing(id: string): Promise<DocumentRecord> {
  return request<DocumentRecord>(`/documents/${id}/index`, {
    method: "POST",
  });
}

export function deleteDocument(id: string): Promise<void> {
  return request<void>(`/documents/${id}`, {
    method: "DELETE",
  });
}

export function askQuestion(
  query: string,
  model: string,
  sessionId: string,
): Promise<ChatResponse> {
  return request<ChatResponse>("/chat", {
    method: "POST",
    body: JSON.stringify({
      query,
      model: model === "all" ? null : model,
      session_id: sessionId,
    }),
  });
}

