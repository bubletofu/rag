// frontend/src/lib/api.ts

const BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export type SourceItem = { source: string; page: number };
export type RetrievalItem = { source: string; page: number; score: number };

export async function listSources(): Promise<string[]> {
  const r = await fetch(`${BASE}/sources`);
  if (!r.ok) return [];
  const data = await r.json();
  return data.sources ?? [];
}

export async function loadDocument(file: File) {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${BASE}/load-document`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function ask(query: string, topK = 6, filter?: string) {
  const body: any = { query, top_k: topK };
  if (filter) body.filters = { source: filter };
  const r = await fetch(`${BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json() as Promise<{
    answer: string;
    sources: SourceItem[];
    retrieval?: RetrievalItem[];
  }>;
}