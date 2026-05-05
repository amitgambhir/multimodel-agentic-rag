import { useRef, useState } from "react";
import {
  FileText, Globe, Image as ImageIcon, FileType, Trash2, Plus, Upload, Loader2,
} from "lucide-react";
import type { Modality, Source, Snapshot } from "../lib/types";
import { addFile, addText, addUrl, deleteSource } from "../lib/api";

const ICONS: Record<Modality, any> = {
  text: FileText, url: Globe, pdf: FileType, image: ImageIcon,
};

const MODALITY_FILTERS: (Modality | "all")[] = ["all", "text", "url", "pdf", "image"];

interface Props {
  snapshot: Snapshot | null;
  onSnapshot: (s: Snapshot) => void;
  citedIds: Set<string>;
}

export default function SourceManager({ snapshot, onSnapshot, citedIds }: Props) {
  const [tab, setTab] = useState<"text" | "url" | "file">("text");
  const [filter, setFilter] = useState<Modality | "all">("all");
  const [textTitle, setTextTitle] = useState("");
  const [textBody, setTextBody] = useState("");
  const [urlVal, setUrlVal] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const sources = snapshot?.sources || [];
  const visible = filter === "all" ? sources : sources.filter((s) => s.modality === filter);

  async function safeRun<T>(fn: () => Promise<T>, after?: () => void) {
    setBusy(true); setErr(null);
    try { const r = await fn(); after?.(); return r; }
    catch (e: any) { setErr(e.message || "Failed"); }
    finally { setBusy(false); }
  }

  async function submitText(e: React.FormEvent) {
    e.preventDefault();
    if (!textTitle.trim() || !textBody.trim()) return;
    await safeRun(async () => {
      const r = await addText(textTitle, textBody);
      onSnapshot(r.space);
    }, () => { setTextTitle(""); setTextBody(""); });
  }

  async function submitUrl(e: React.FormEvent) {
    e.preventDefault();
    if (!urlVal.trim()) return;
    await safeRun(async () => {
      const r = await addUrl(urlVal);
      onSnapshot(r.space);
    }, () => setUrlVal(""));
  }

  async function handleFiles(files: FileList | null) {
    if (!files || files.length === 0) return;
    for (const f of Array.from(files)) {
      await safeRun(async () => {
        const r = await addFile(f, f.name);
        onSnapshot(r.space);
      });
    }
  }

  async function remove(id: string) {
    await safeRun(async () => {
      const r = await deleteSource(id);
      onSnapshot(r.space);
    });
  }

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex items-center justify-between px-4 pt-4 pb-2">
        <h2 className="text-sm font-semibold tracking-wide uppercase text-[color:var(--color-fg-muted)]">
          Sources
          <span className="ml-2 text-xs font-normal text-[color:var(--color-fg-muted)]/70">
            {sources.length}
          </span>
        </h2>
      </div>

      {/* Add tabs */}
      <div className="px-4">
        <div className="flex gap-1 rounded-lg bg-[color:var(--color-bg-soft)] p-1 border border-[color:var(--color-border)]">
          {(["text", "url", "file"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`flex-1 text-xs font-medium px-2 py-1.5 rounded-md transition ${
                tab === t
                  ? "bg-[color:var(--color-panel-2)] text-[color:var(--color-fg)] shadow-sm"
                  : "text-[color:var(--color-fg-muted)] hover:text-[color:var(--color-fg)]"
              }`}
            >
              {t === "text" ? "Text" : t === "url" ? "URL" : "File"}
            </button>
          ))}
        </div>
      </div>

      <div className="px-4 pt-3">
        {tab === "text" && (
          <form onSubmit={submitText} className="space-y-2">
            <input
              value={textTitle}
              onChange={(e) => setTextTitle(e.target.value)}
              placeholder="Title"
              className="w-full bg-[color:var(--color-bg-soft)] border border-[color:var(--color-border)] rounded-md px-3 py-2 text-sm outline-none focus:border-[color:var(--color-accent)]"
            />
            <textarea
              value={textBody}
              onChange={(e) => setTextBody(e.target.value)}
              placeholder="Paste text…"
              rows={4}
              className="w-full bg-[color:var(--color-bg-soft)] border border-[color:var(--color-border)] rounded-md px-3 py-2 text-sm outline-none resize-none focus:border-[color:var(--color-accent)] scrollbar-thin"
            />
            <button
              type="submit"
              disabled={busy || !textTitle.trim() || !textBody.trim()}
              className="w-full inline-flex items-center justify-center gap-1.5 bg-[color:var(--color-accent)]/90 hover:bg-[color:var(--color-accent)] disabled:opacity-40 text-black font-medium text-sm rounded-md py-2 transition"
            >
              {busy ? <Loader2 size={14} className="animate-spin" /> : <Plus size={14} />} Add text
            </button>
          </form>
        )}

        {tab === "url" && (
          <form onSubmit={submitUrl} className="space-y-2">
            <input
              value={urlVal}
              onChange={(e) => setUrlVal(e.target.value)}
              placeholder="https://example.com/article"
              className="w-full bg-[color:var(--color-bg-soft)] border border-[color:var(--color-border)] rounded-md px-3 py-2 text-sm outline-none focus:border-[color:var(--color-accent)]"
            />
            <button
              type="submit"
              disabled={busy || !urlVal.trim()}
              className="w-full inline-flex items-center justify-center gap-1.5 bg-[color:var(--color-accent)]/90 hover:bg-[color:var(--color-accent)] disabled:opacity-40 text-black font-medium text-sm rounded-md py-2 transition"
            >
              {busy ? <Loader2 size={14} className="animate-spin" /> : <Globe size={14} />} Fetch & index
            </button>
          </form>
        )}

        {tab === "file" && (
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFiles(e.dataTransfer.files); }}
            onClick={() => fileInputRef.current?.click()}
            className={`cursor-pointer rounded-lg border-2 border-dashed transition px-4 py-6 flex flex-col items-center justify-center gap-1.5 text-center ${
              dragOver
                ? "border-[color:var(--color-accent)] bg-[color:var(--color-accent)]/10"
                : "border-[color:var(--color-border)] hover:border-[color:var(--color-accent)]/60 bg-[color:var(--color-bg-soft)]"
            }`}
          >
            {busy ? <Loader2 size={18} className="animate-spin" /> : <Upload size={18} />}
            <div className="text-xs font-medium">Drop PDFs or images</div>
            <div className="text-[11px] text-[color:var(--color-fg-muted)]">or click to choose</div>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,image/*"
              multiple
              hidden
              onChange={(e) => handleFiles(e.target.files)}
            />
          </div>
        )}

        {err && (
          <div className="mt-2 text-[11px] text-[color:var(--color-warn)] bg-[color:var(--color-warn)]/10 border border-[color:var(--color-warn)]/30 rounded-md px-2 py-1">
            {err}
          </div>
        )}
      </div>

      {/* Filters */}
      <div className="px-4 pt-4 pb-2 flex items-center gap-1.5 flex-wrap">
        {MODALITY_FILTERS.map((m) => (
          <button
            key={m}
            onClick={() => setFilter(m)}
            className={`text-[11px] px-2 py-0.5 rounded-full border transition ${
              filter === m
                ? "border-[color:var(--color-accent)] text-[color:var(--color-accent)] bg-[color:var(--color-accent)]/10"
                : "border-[color:var(--color-border)] text-[color:var(--color-fg-muted)] hover:text-[color:var(--color-fg)]"
            }`}
          >
            {m}
          </button>
        ))}
      </div>

      {/* List */}
      <div className="flex-1 min-h-0 overflow-y-auto px-2 pb-4 scrollbar-thin">
        {visible.length === 0 && (
          <div className="text-xs text-[color:var(--color-fg-muted)] px-3 py-6 text-center">
            No sources yet. Add text, a URL, a PDF, or an image.
          </div>
        )}
        <ul className="space-y-1.5">
          {visible.map((s) => (
            <SourceCard
              key={s.id}
              source={s}
              cited={citedIds.has(s.id)}
              onRemove={() => remove(s.id)}
            />
          ))}
        </ul>
      </div>
    </div>
  );
}

function SourceCard({ source, cited, onRemove }: { source: Source; cited: boolean; onRemove: () => void }) {
  const Icon = ICONS[source.modality];
  const isImage = source.modality === "image" && (source.preview || "").startsWith("data:");

  return (
    <li
      className={`group rounded-lg border px-3 py-2 transition fade-in ${
        cited
          ? "border-[color:var(--color-accent)]/70 bg-[color:var(--color-accent)]/5"
          : "border-[color:var(--color-border)] bg-[color:var(--color-bg-soft)] hover:border-[color:var(--color-border)]/60"
      }`}
    >
      <div className="flex items-start gap-2">
        {isImage ? (
          <img src={source.preview} alt="" className="w-9 h-9 rounded object-cover border border-[color:var(--color-border)]" />
        ) : (
          <div className="w-9 h-9 flex items-center justify-center rounded bg-[color:var(--color-panel-2)] text-[color:var(--color-fg-muted)]">
            <Icon size={16} />
          </div>
        )}
        <div className="flex-1 min-w-0">
          <div className="text-xs font-medium truncate">{source.title}</div>
          <div className="text-[10px] uppercase tracking-wider text-[color:var(--color-fg-muted)]">
            {source.modality} · {source.chunk_count} chunk{source.chunk_count === 1 ? "" : "s"}
            {cited && (
              <span className="ml-1 text-[color:var(--color-accent)]">· cited</span>
            )}
          </div>
          {source.preview && !isImage && (
            <div className="text-[11px] text-[color:var(--color-fg-muted)] line-clamp-2 mt-0.5">
              {source.preview}
            </div>
          )}
        </div>
        <button
          onClick={onRemove}
          className="opacity-0 group-hover:opacity-100 text-[color:var(--color-fg-muted)] hover:text-[color:var(--color-warn)] transition"
          title="Remove"
        >
          <Trash2 size={14} />
        </button>
      </div>
    </li>
  );
}
