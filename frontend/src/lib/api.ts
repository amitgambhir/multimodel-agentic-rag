import type { Snapshot, StreamEvent, Provider } from "./types";

const API = (import.meta as any).env?.VITE_API_URL || "/api";

export async function getHealth() {
  const r = await fetch(`${API}/health`);
  if (!r.ok) throw new Error("health check failed");
  return r.json();
}

export async function getSpace(): Promise<Snapshot> {
  const r = await fetch(`${API}/space`);
  if (!r.ok) throw new Error("space failed");
  return r.json();
}

export async function addText(title: string, text: string) {
  const r = await fetch(`${API}/sources/text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, text }),
  });
  if (!r.ok) throw new Error((await r.json()).detail || "Add text failed");
  return r.json();
}

export async function addUrl(url: string, title?: string) {
  const r = await fetch(`${API}/sources/url`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, title }),
  });
  if (!r.ok) throw new Error((await r.json()).detail || "Add URL failed");
  return r.json();
}

export async function addFile(file: File, title?: string, notes?: string) {
  const fd = new FormData();
  fd.append("file", file);
  if (title) fd.append("title", title);
  if (notes) fd.append("notes", notes);
  const r = await fetch(`${API}/sources/file`, { method: "POST", body: fd });
  if (!r.ok) throw new Error((await r.json()).detail || "Add file failed");
  return r.json();
}

export async function deleteSource(id: string) {
  const r = await fetch(`${API}/sources/${id}`, { method: "DELETE" });
  if (!r.ok) throw new Error("Delete failed");
  return r.json();
}

export interface AskOptions {
  question: string;
  provider: Provider;
  topK: number;
  maxHops: number;
  onEvent: (e: StreamEvent) => void;
  signal?: AbortSignal;
}

export async function ask({ question, provider, topK, maxHops, onEvent, signal }: AskOptions) {
  const r = await fetch(`${API}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, provider, top_k: topK, max_hops: maxHops }),
    signal,
  });
  if (!r.ok || !r.body) {
    const detail = await r.text().catch(() => "");
    throw new Error(`Ask failed: ${r.status} ${detail}`);
  }
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop() || "";
    for (const part of parts) {
      const line = part.trim();
      if (!line.startsWith("data:")) continue;
      const json = line.slice(5).trim();
      if (!json) continue;
      try {
        onEvent(JSON.parse(json));
      } catch {
        // ignore malformed
      }
    }
  }
}
