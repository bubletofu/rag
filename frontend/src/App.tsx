// frontend/src/App.tsx

import { useEffect, useMemo, useState } from "react";

type Source = { source: string; page: number };
type Retrieval = { source: string; page: number; score: number };

const API = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export default function App() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(6);
  const [sources, setSources] = useState<string[]>([]);
  const [selectedSource, setSelectedSource] = useState("");
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState<Source[]>([]);
  const [retrieval, setRetrieval] = useState<Retrieval[]>([]);
  const [loading, setLoading] = useState(false);
  const [idxLoading, setIdxLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${API}/sources`)
      .then((r) => r.json())
      .then((d) => setSources(d.sources ?? []))
      .catch(() => setSources([]));
  }, []);

  const canAsk = useMemo(() => query.trim().length > 0 && !loading, [query, loading]);

  const ask = async () => {
    try {
      setLoading(true);
      setError("");
      setAnswer("");
      setCitations([]);
      setRetrieval([]);
      const body: any = { query, top_k: topK };
      if (selectedSource) body.filters = { source: selectedSource };
      const r = await fetch(`${API}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      setAnswer(data.answer || "");
      setCitations(data.sources || []);
      setRetrieval(data.retrieval || []);
    } catch (e: any) {
      setError(e?.message || "Request failed");
    } finally {
      setLoading(false);
    }
  };

  const upload = async (file?: File) => {
    if (!file) return;
    try {
      setIdxLoading(true);
      setError("");
      const fd = new FormData();
      fd.append("file", file);
      const r = await fetch(`${API}/load-document`, { method: "POST", body: fd });
      if (!r.ok) throw new Error(await r.text());
      await fetch(`${API}/sources`)
        .then((r) => r.json())
        .then((d) => setSources(d.sources ?? []));
    } catch (e: any) {
      setError(e?.message || "Upload failed");
    } finally {
      setIdxLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 960, margin: "40px auto", padding: 16, fontFamily: "Inter, ui-sans-serif" }}>
      <h1>Ask question here</h1>

      <div style={{ display: "grid", gridTemplateColumns: "1fr auto auto", gap: 8, marginBottom: 12 }}>
        <select
          value={selectedSource}
          onChange={(e) => setSelectedSource(e.target.value)}
          style={{ padding: 8, borderRadius: 8, border: "1px solid #ddd" }}
        >
          <option value="">All sources</option>
          {sources.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>

        <input
          type="number"
          min={1}
          max={20}
          value={topK}
          onChange={(e) => setTopK(Number(e.target.value))}
          style={{ width: 90, padding: 8, borderRadius: 8, border: "1px solid #ddd" }}
        />

        <label
          style={{ padding: "8px 12px", borderRadius: 8, border: "1px solid #ddd", background: idxLoading ? "#eee" : "#fafafa", cursor: idxLoading ? "not-allowed" : "pointer" }}
        >
          <input type="file" accept="application/pdf" style={{ display: "none" }} onChange={(e) => upload(e.target.files?.[0] || undefined)} disabled={idxLoading} />
          {idxLoading ? "Indexing…" : "Upload PDF"}
        </label>
      </div>

      <textarea
        rows={5}
        placeholder="How may i help you?"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        style={{ width: "100%", padding: 12, borderRadius: 12, border: "1px solid #ddd" }}
      />

      <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
        <button onClick={ask} disabled={!canAsk} style={{ padding: "10px 16px", borderRadius: 10, border: "1px solid #0a7", background: canAsk ? "#0a7" : "#7ac6b5", color: "white" }}>
          {loading ? "Thinking…" : "Ask"}
        </button>
        {error && <span style={{ color: "#b00", alignSelf: "center" }}>{error}</span>}
      </div>

      {answer && (
        <section style={{ marginTop: 24 }}>
          <h2>Answer</h2>
          <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.6, background: "#fcfcfc", border: "1px solid #eee", padding: 16, borderRadius: 12 }}>{answer}</div>
        </section>
      )}

      {citations.length > 0 && (
        <section style={{ marginTop: 24 }}>
          <h3>Sources</h3>
          <ul style={{ marginTop: 0 }}>
            {citations.map((c, i) => (
              <li key={`${c.source}-${c.page}-${i}`}>
                <code>{c.source}</code> (p.{c.page})
              </li>
            ))}
          </ul>
        </section>
      )}

      {retrieval.length > 0 && (
        <section style={{ marginTop: 16 }}>
          <h4 style={{ color: "#666" }}>Retrieval (debug)</h4>
          <div style={{ fontFamily: "ui-monospace", fontSize: 13, background: "#fbfbfb", border: "1px solid #eee", padding: 12, borderRadius: 8 }}>
            {retrieval.map((r, i) => (
              <div key={`${r.source}-${r.page}-${i}`}>
                {r.source} (p.{r.page}) — score {r.score.toFixed(4)}
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}