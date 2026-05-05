import { useEffect, useRef, useState } from "react";
import { Send, Loader2, Square, Sparkles, Zap } from "lucide-react";
import { ask } from "../lib/api";
import type { Match, Provider, Snapshot, TraceStep, Usage } from "../lib/types";

interface Props {
  provider: Provider;
  topK: number;
  maxHops: number;
  onSnapshot: (s: Snapshot) => void;
  onMatches: (m: Match[]) => void;
  onTrace: (t: TraceStep[], usage: Usage | null) => void;
}

export default function AskPanel({
  provider, topK, maxHops, onSnapshot, onMatches, onTrace,
}: Props) {
  const [q, setQ] = useState("");
  const [answer, setAnswer] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [matches, setMatches] = useState<Match[]>([]);
  const [usage, setUsage] = useState<Usage | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [hovered, setHovered] = useState<number | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const startRef = useRef<number>(0);

  function stop() { abortRef.current?.abort(); }

  async function submit(e?: React.FormEvent) {
    e?.preventDefault();
    if (!q.trim() || streaming) return;

    setAnswer(""); setMatches([]); setUsage(null); setErr(null); setLatencyMs(null);
    setStreaming(true);
    startRef.current = performance.now();

    const ctrl = new AbortController();
    abortRef.current = ctrl;

    const collectedTrace: TraceStep[] = [];
    let collectedMatches: Match[] = [];
    let collectedUsage: Usage | null = null;

    try {
      await ask({
        question: q,
        provider,
        topK,
        maxHops,
        signal: ctrl.signal,
        onEvent: (ev) => {
          switch (ev.type) {
            case "trace":
              collectedTrace.push({ step: ev.step, detail: ev.detail });
              onTrace([...collectedTrace], collectedUsage);
              break;
            case "answer_delta":
              setAnswer((a) => a + ev.delta);
              break;
            case "reset_answer":
              setAnswer("");
              break;
            case "tool":
              collectedTrace.push({ step: ev.name, detail: ev.result_summary });
              onTrace([...collectedTrace], collectedUsage);
              break;
            case "usage":
              collectedUsage = {
                input_tokens: ev.input_tokens,
                output_tokens: ev.output_tokens,
                cache_read_tokens: ev.cache_read_tokens,
                cache_creation_tokens: ev.cache_creation_tokens,
              };
              setUsage(collectedUsage);
              onTrace([...collectedTrace], collectedUsage);
              break;
            case "done":
              collectedMatches = ev.matches;
              setMatches(ev.matches);
              onMatches(ev.matches);
              onSnapshot(ev.space);
              setLatencyMs(Math.round(performance.now() - startRef.current));
              break;
            case "error":
              setErr(ev.message);
              break;
          }
        },
      });
    } catch (e: any) {
      if (e.name !== "AbortError") setErr(e.message || "Stream failed");
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  }

  return (
    <div className="flex flex-col h-full min-h-0">
      <form onSubmit={submit} className="px-4 pt-4 pb-3 border-b border-[color:var(--color-border)]">
        <div className="relative">
          <textarea
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit();
            }}
            placeholder="Ask a question over your sources… (⌘/Ctrl+Enter to send)"
            rows={2}
            className="w-full bg-[color:var(--color-bg-soft)] border border-[color:var(--color-border)] rounded-lg px-3.5 py-3 pr-24 text-sm outline-none resize-none focus:border-[color:var(--color-accent)] scrollbar-thin"
          />
          <div className="absolute right-2 top-2 flex items-center gap-1.5">
            {streaming ? (
              <button
                type="button"
                onClick={stop}
                className="inline-flex items-center gap-1 text-xs px-2.5 py-1.5 rounded-md bg-[color:var(--color-warn)]/15 text-[color:var(--color-warn)] hover:bg-[color:var(--color-warn)]/25"
              >
                <Square size={12} /> Stop
              </button>
            ) : (
              <button
                type="submit"
                disabled={!q.trim()}
                className="inline-flex items-center gap-1 text-xs px-3 py-1.5 rounded-md bg-[color:var(--color-accent)]/90 hover:bg-[color:var(--color-accent)] text-black font-medium disabled:opacity-40"
              >
                <Send size={12} /> Ask
              </button>
            )}
          </div>
        </div>
        <div className="flex items-center gap-3 mt-2 text-[11px] text-[color:var(--color-fg-muted)]">
          <span className="inline-flex items-center gap-1">
            <Sparkles size={11} className="text-[color:var(--color-accent)]" /> {provider}
          </span>
          <span>top_k={topK}</span>
          <span>hops={maxHops}</span>
          {latencyMs !== null && (
            <span className="inline-flex items-center gap-1">
              <Zap size={11} /> {latencyMs} ms
            </span>
          )}
          {usage && (
            <span>
              tokens in/out: {usage.input_tokens}/{usage.output_tokens}
              {usage.cache_read_tokens > 0 && (
                <span className="text-[color:var(--color-success)]">
                  {" "}· cache hit {usage.cache_read_tokens}
                </span>
              )}
            </span>
          )}
        </div>
      </form>

      <div className="flex-1 min-h-0 overflow-y-auto scrollbar-thin">
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-0 h-full">
          {/* Answer */}
          <div className="lg:col-span-3 p-5 border-r border-[color:var(--color-border)]">
            <h3 className="text-[11px] font-semibold uppercase tracking-wider text-[color:var(--color-fg-muted)] mb-2">
              Answer
            </h3>
            {!answer && !streaming && (
              <div className="text-sm text-[color:var(--color-fg-muted)]">
                Ask a question to see a grounded answer with citations.
              </div>
            )}
            {streaming && !answer && (
              <div className="inline-flex items-center gap-2 text-sm text-[color:var(--color-fg-muted)]">
                <Loader2 size={14} className="animate-spin" /> Thinking…
              </div>
            )}
            {answer && (
              <div className="text-[15px] leading-relaxed whitespace-pre-wrap fade-in">
                {answer}
                {streaming && <span className="inline-block w-2 h-4 ml-0.5 align-middle bg-[color:var(--color-accent)] pulse-dot" />}
              </div>
            )}
            {err && (
              <div className="mt-3 text-xs text-[color:var(--color-warn)] bg-[color:var(--color-warn)]/10 border border-[color:var(--color-warn)]/30 rounded-md px-3 py-2">
                {err}
              </div>
            )}
          </div>

          {/* Citations */}
          <div className="lg:col-span-2 p-5">
            <h3 className="text-[11px] font-semibold uppercase tracking-wider text-[color:var(--color-fg-muted)] mb-2">
              Citations <span className="text-[color:var(--color-fg-muted)]/60">{matches.length > 0 && `· ${matches.length}`}</span>
            </h3>
            <ul className="space-y-2">
              {matches.map((m, i) => (
                <li
                  key={`${m.source_id}-${m.chunk_index}`}
                  onMouseEnter={() => setHovered(i)}
                  onMouseLeave={() => setHovered(null)}
                  className={`rounded-lg border px-3 py-2 fade-in transition ${
                    hovered === i
                      ? "border-[color:var(--color-accent)] bg-[color:var(--color-accent)]/10"
                      : "border-[color:var(--color-border)] bg-[color:var(--color-bg-soft)]"
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="text-xs font-medium truncate">
                      <span className="text-[color:var(--color-accent)] mr-1">[{i + 1}]</span>
                      {m.source}
                    </div>
                    <span className="text-[10px] uppercase tracking-wider text-[color:var(--color-fg-muted)]">
                      {m.modality} · {m.score.toFixed(3)}
                    </span>
                  </div>
                  <div className="text-[11px] text-[color:var(--color-fg-muted)] mt-1 line-clamp-3">
                    {m.snippet}
                  </div>
                </li>
              ))}
              {matches.length === 0 && !streaming && (
                <li className="text-xs text-[color:var(--color-fg-muted)]">No citations yet.</li>
              )}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
