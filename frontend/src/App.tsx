import { useEffect, useMemo, useState } from "react";
import { Boxes, ChevronRight, Database, Sparkles } from "lucide-react";
import SourceManager from "./components/SourceManager";
import AskPanel from "./components/AskPanel";
import EmbeddingView from "./components/EmbeddingView";
import TraceDrawer from "./components/TraceDrawer";
import { getHealth, getSpace } from "./lib/api";
import type { Match, Provider, Snapshot, TraceStep, Usage } from "./lib/types";

export default function App() {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [matches, setMatches] = useState<Match[]>([]);
  const [trace, setTrace] = useState<TraceStep[]>([]);
  const [usage, setUsage] = useState<Usage | null>(null);
  const [providers, setProviders] = useState<Provider[]>([]);
  const [provider, setProvider] = useState<Provider>("claude");
  const [topK, setTopK] = useState(6);
  const [maxHops, setMaxHops] = useState(2);
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const [embeddingProvider, setEmbeddingProvider] = useState<string>("");

  useEffect(() => {
    (async () => {
      try {
        const h = await getHealth();
        setHealthy(true);
        const provs = (h.providers || []) as Provider[];
        setProviders(provs);
        if (provs.length && !provs.includes(provider)) setProvider(provs[0]);
        setEmbeddingProvider(h.embedding_provider || "");
        const s = await getSpace();
        setSnapshot(s);
      } catch {
        setHealthy(false);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const citedIds = useMemo(() => new Set(matches.map((m) => m.source_id)), [matches]);

  return (
    <div className="h-full flex flex-col gradient-bg">
      <Header
        provider={provider}
        setProvider={setProvider}
        providers={providers}
        topK={topK}
        setTopK={setTopK}
        maxHops={maxHops}
        setMaxHops={setMaxHops}
        healthy={healthy}
        embeddingProvider={embeddingProvider}
      />

      <div className="flex-1 grid grid-cols-12 gap-3 px-3 pb-3 min-h-0">
        {/* Left: sources */}
        <aside className="col-span-12 md:col-span-3 glass rounded-xl min-h-0">
          <SourceManager
            snapshot={snapshot}
            onSnapshot={setSnapshot}
            citedIds={citedIds}
          />
        </aside>

        {/* Center: ask + answer */}
        <main className="col-span-12 md:col-span-6 glass rounded-xl min-h-0">
          <AskPanel
            provider={provider}
            topK={topK}
            maxHops={maxHops}
            onSnapshot={setSnapshot}
            onMatches={setMatches}
            onTrace={(t, u) => { setTrace(t); setUsage(u); }}
          />
        </main>

        {/* Right: 3D + trace */}
        <aside className="col-span-12 md:col-span-3 grid grid-rows-2 gap-3 min-h-0">
          <div className="glass rounded-xl overflow-hidden min-h-0 relative">
            <div className="absolute top-2 left-3 z-10 inline-flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-[color:var(--color-fg-muted)]">
              <Boxes size={11} /> Embedding space
            </div>
            <EmbeddingView snapshot={snapshot} citedIds={citedIds} />
          </div>
          <div className="glass rounded-xl min-h-0">
            <TraceDrawer trace={trace} usage={usage} />
          </div>
        </aside>
      </div>
    </div>
  );
}

interface HeaderProps {
  provider: Provider;
  setProvider: (p: Provider) => void;
  providers: Provider[];
  topK: number;
  setTopK: (n: number) => void;
  maxHops: number;
  setMaxHops: (n: number) => void;
  healthy: boolean | null;
  embeddingProvider: string;
}

function Header({
  provider, setProvider, providers, topK, setTopK,
  maxHops, setMaxHops, healthy, embeddingProvider,
}: HeaderProps) {
  return (
    <header className="flex items-center justify-between px-4 py-3">
      <div className="flex items-center gap-2.5">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[color:var(--color-accent)] to-[color:var(--color-accent-2)] flex items-center justify-center text-black">
          <Sparkles size={16} />
        </div>
        <div>
          <div className="text-sm font-semibold leading-tight">Multi-LLM Agentic RAG</div>
          <div className="text-[11px] text-[color:var(--color-fg-muted)] flex items-center gap-1">
            <Database size={10} /> {embeddingProvider || "no embeddings"}
            <ChevronRight size={10} className="opacity-60" />
            agent · {maxHops} hop{maxHops === 1 ? "" : "s"}
          </div>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <div className="hidden md:flex items-center gap-1.5 text-[11px] text-[color:var(--color-fg-muted)]">
          <span>top_k</span>
          <input
            type="number"
            min={1}
            max={12}
            value={topK}
            onChange={(e) => setTopK(Math.max(1, Math.min(12, Number(e.target.value) || 6)))}
            className="w-12 bg-[color:var(--color-bg-soft)] border border-[color:var(--color-border)] rounded px-1.5 py-0.5 text-center text-xs"
          />
        </div>
        <div className="hidden md:flex items-center gap-1.5 text-[11px] text-[color:var(--color-fg-muted)]">
          <span>hops</span>
          <select
            value={maxHops}
            onChange={(e) => setMaxHops(Number(e.target.value))}
            className="bg-[color:var(--color-bg-soft)] border border-[color:var(--color-border)] rounded px-1.5 py-0.5 text-xs"
          >
            <option value={1}>1</option>
            <option value={2}>2</option>
            <option value={3}>3</option>
          </select>
        </div>

        <ProviderPicker provider={provider} providers={providers} onPick={setProvider} />

        <div
          className={`text-[11px] uppercase tracking-wider px-2 py-1 rounded-full border ${
            healthy === false
              ? "border-[color:var(--color-warn)]/50 text-[color:var(--color-warn)]"
              : healthy === null
              ? "border-[color:var(--color-border)] text-[color:var(--color-fg-muted)]"
              : "border-[color:var(--color-success)]/50 text-[color:var(--color-success)]"
          }`}
          title={healthy === false ? "Backend not reachable" : ""}
        >
          {healthy === false ? "offline" : healthy === null ? "…" : "online"}
        </div>
      </div>
    </header>
  );
}

function ProviderPicker({
  provider, providers, onPick,
}: { provider: Provider; providers: Provider[]; onPick: (p: Provider) => void }) {
  const all: Provider[] = ["claude", "gemini"];
  return (
    <div className="inline-flex rounded-lg overflow-hidden border border-[color:var(--color-border)] bg-[color:var(--color-bg-soft)]">
      {all.map((p) => {
        const enabled = providers.includes(p);
        const active = provider === p;
        return (
          <button
            key={p}
            disabled={!enabled}
            onClick={() => onPick(p)}
            className={`text-xs px-3 py-1.5 transition ${
              active
                ? "bg-[color:var(--color-accent)] text-black font-semibold"
                : enabled
                ? "text-[color:var(--color-fg-muted)] hover:text-[color:var(--color-fg)]"
                : "text-[color:var(--color-fg-muted)]/40 cursor-not-allowed"
            }`}
            title={!enabled ? `${p.toUpperCase()} key not configured` : ""}
          >
            {p}
          </button>
        );
      })}
    </div>
  );
}
