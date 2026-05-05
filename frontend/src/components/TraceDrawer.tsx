import { Activity } from "lucide-react";
import type { TraceStep, Usage } from "../lib/types";

interface Props {
  trace: TraceStep[];
  usage: Usage | null;
}

export default function TraceDrawer({ trace, usage }: Props) {
  return (
    <div className="h-full flex flex-col min-h-0">
      <div className="px-4 py-3 border-b border-[color:var(--color-border)] flex items-center gap-2">
        <Activity size={14} className="text-[color:var(--color-accent)]" />
        <h3 className="text-sm font-semibold tracking-wide uppercase text-[color:var(--color-fg-muted)]">
          Trace
        </h3>
      </div>
      <div className="flex-1 min-h-0 overflow-y-auto scrollbar-thin px-4 py-3">
        {trace.length === 0 && (
          <div className="text-xs text-[color:var(--color-fg-muted)]">
            Ask a question to see the agent's retrieval and reasoning trace.
          </div>
        )}
        <ol className="space-y-2">
          {trace.map((t, i) => (
            <li key={i} className="text-[12px] fade-in">
              <div className="flex items-start gap-2">
                <span className="mt-1 w-1.5 h-1.5 rounded-full bg-[color:var(--color-accent)]" />
                <div className="flex-1 min-w-0">
                  <div className="text-[10px] uppercase tracking-wider text-[color:var(--color-fg-muted)]">
                    {t.step}
                  </div>
                  <div className="text-[12px] text-[color:var(--color-fg)] break-words">{t.detail}</div>
                </div>
              </div>
            </li>
          ))}
        </ol>
      </div>
      {usage && (
        <div className="px-4 py-2 border-t border-[color:var(--color-border)] text-[11px] text-[color:var(--color-fg-muted)]">
          tokens in/out: {usage.input_tokens}/{usage.output_tokens}
          {usage.cache_read_tokens > 0 && (
            <span className="text-[color:var(--color-success)]"> · cache read {usage.cache_read_tokens}</span>
          )}
          {usage.cache_creation_tokens > 0 && (
            <span> · cache write {usage.cache_creation_tokens}</span>
          )}
        </div>
      )}
    </div>
  );
}
