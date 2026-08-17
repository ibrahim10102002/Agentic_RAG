"use client";

import { useState, useRef, useEffect } from "react";
import {
  Search, ChevronDown, ChevronUp, Loader2,
  GitBranch, RefreshCw, CheckCircle, AlertCircle,
  Zap, Terminal, TrendingUp
} from "lucide-react";

// ── Types ──────────────────────────────────────────────
interface TraceStep {
  step: string;
  decision: Record<string, unknown>;
}
interface CitedChunk {
  citation_number: number;
  company: string;
  ticker: string;
  section: string;
  text_snippet: string;
}
interface Source {
  chunk_id: string;
  company: string;
  ticker: string;
  section: string;
  text: string;
  rerank_score: number;
}
interface ConfidenceResult {
  is_sufficient: boolean;
  confidence: number;
  signals: { chunk_count: number; top_score: number; average_score: number; score_spread: number };
  failures: string[];
  recommendation: string;
}
interface QueryResult {
  query: string;
  answer: string;
  cited_chunks: CitedChunk[];
  sources: Source[];
  confidence: ConfidenceResult;
  trace: TraceStep[];
  elapsed_sec: number;
  attempts: number;
}

// ── Constants ──────────────────────────────────────────
const COMPANIES = [
  { name: "Apple",      ticker: "AAPL" },
  { name: "Microsoft",  ticker: "MSFT" },
  { name: "Alphabet",   ticker: "GOOGL" },
  { name: "Meta",       ticker: "META" },
  { name: "Nvidia",     ticker: "NVDA" },
  { name: "Amazon",     ticker: "AMZN" },
  { name: "Tesla",      ticker: "TSLA" },
  { name: "Netflix",    ticker: "NFLX" },
  { name: "Salesforce", ticker: "CRM" },
  { name: "AMD",        ticker: "AMD" },
];

const SECTION_STYLES: Record<string, string> = {
  risk_factors: "border-red-500/40 text-red-400 bg-red-500/10",
  mda:          "border-blue-500/40 text-blue-400 bg-blue-500/10",
  financials:   "border-emerald-500/40 text-emerald-400 bg-emerald-500/10",
  business:     "border-purple-500/40 text-purple-400 bg-purple-500/10",
  general:      "border-slate-500/40 text-slate-400 bg-slate-500/10",
};

const SAMPLE_QUERIES = [
  "What are Nvidia's biggest risk factors?",
  "How did Apple's revenue grow last year?",
  "What cybersecurity threats does Microsoft face?",
  "Compare Tesla and AMD's business models",
];

// ── Ticker Tape ────────────────────────────────────────
function TickerTape() {
  const items = [...COMPANIES, ...COMPANIES]; // duplicate for seamless loop
  return (
    <div className="w-full overflow-hidden border-b border-amber-500/20 bg-amber-500/5">
      <div className="flex animate-marquee gap-0">
        {items.map((c, i) => (
          <span key={i} className="flex items-center gap-3 px-6 py-2 text-xs whitespace-nowrap border-r border-amber-500/10">
            <span className="font-mono font-bold text-amber-400">{c.ticker}</span>
            <span className="text-slate-400">{c.name}</span>
            <span className="text-slate-600">10-K</span>
          </span>
        ))}
      </div>
    </div>
  );
}

// ── Section Badge ──────────────────────────────────────
function SectionBadge({ section }: { section: string }) {
  return (
    <span className={`text-xs px-2 py-0.5 rounded border font-mono ${SECTION_STYLES[section] ?? SECTION_STYLES.general}`}>
      {section.replace("_", "_")}
    </span>
  );
}

// ── Answer Text ────────────────────────────────────────
function AnswerText({ text }: { text: string }) {
  const parts = text.split(/(\[\d+\])/g);
  return (
    <p className="text-slate-200 leading-relaxed text-sm whitespace-pre-wrap">
      {parts.map((part, i) => {
        const match = part.match(/^\[(\d+)\]$/);
        if (match) {
          return (
            <sup key={i}>
              <span className="inline-flex items-center justify-center w-4 h-4 text-xs font-bold bg-amber-500 text-black rounded mx-0.5 font-mono">
                {match[1]}
              </span>
            </sup>
          );
        }
        return <span key={i}>{part}</span>;
      })}
    </p>
  );
}

// ── Trace Step ─────────────────────────────────────────
function TraceStep({ step, index }: { step: TraceStep; index: number }) {
  const [open, setOpen] = useState(index <= 1);
  const d = step.decision;

  const isRetry    = step.step.startsWith("retry_");
  const isRetrieve = step.step.startsWith("retrieve_attempt_");
  const isSufficient   = isRetrieve && Boolean(d.is_sufficient);
  const isInsufficient = isRetrieve && !d.is_sufficient;

  const prefix =
    step.step === "route"       ? "01 ROUTE" :
    step.step === "reformulate" ? "02 REWRITE" :
    isRetrieve ? `0${index} FETCH` :
    isRetry    ? `↻  RETRY` :
    step.step === "generate"    ? "✓  GENERATE" : step.step.toUpperCase();

  const prefixColor =
    isSufficient   ? "text-emerald-400" :
    isInsufficient ? "text-amber-400"   :
    isRetry        ? "text-red-400"     :
    step.step === "generate" ? "text-emerald-400" : "text-slate-400";

  return (
    <div className="border-l-2 border-slate-700 pl-3 py-1">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-3 text-left group"
      >
        <span className={`font-mono text-xs font-bold min-w-24 ${prefixColor}`}>
          {prefix}
        </span>
        <span className="text-xs text-slate-500 flex-1 truncate">
          {isRetrieve
            ? `${d.chunks_returned} chunks · conf ${d.confidence}`
            : step.step === "route"
              ? `${d.query_type} → [${(d.sections as string[]).join(", ")}]`
              : step.step === "reformulate"
                ? (d.needs_reformulation ? "rewritten" : "no change")
                : step.step === "generate"
                  ? `${d.chunks_used} chunks → ${d.citations_used} citations`
                  : String(d.filter_loosened ?? "")}
        </span>
        {isRetrieve && (
          isSufficient
            ? <CheckCircle className="w-3 h-3 text-emerald-400 flex-shrink-0" />
            : <AlertCircle className="w-3 h-3 text-amber-400 flex-shrink-0" />
        )}
        {open
          ? <ChevronUp className="w-3 h-3 text-slate-600 flex-shrink-0" />
          : <ChevronDown className="w-3 h-3 text-slate-600 flex-shrink-0" />}
      </button>

      {open && (
        <div className="mt-2 ml-2 font-mono text-xs space-y-1 border-l border-slate-800 pl-3">
          {step.step === "route" && (<>
            <LogLine k="type"     v={String(d.query_type)} />
            <LogLine k="sections" v={(d.sections as string[]).join(", ") || "—"} />
            <LogLine k="tickers"  v={(d.companies as string[]).join(", ") || "all"} />
            <LogLine k="rewrite"  v={d.needs_reformulation ? "yes" : "no"} color={d.needs_reformulation ? "text-amber-400" : "text-emerald-400"} />
            <LogLine k="reason"   v={String(d.reasoning)} />
          </>)}

          {step.step === "reformulate" && (<>
            <LogLine k="original" v={String(d.original_query)} />
            <LogLine k="rewrite"  v={String(d.reformulated_query)} color="text-amber-300" />
            <LogLine k="reason"   v={String(d.reason)} />
          </>)}

          {isRetrieve && (<>
            <LogLine k="query"  v={String(d.query_used)} />
            <LogLine k="filter" v={String(d.filter_used)} />
            <LogLine k="chunks" v={String(d.chunks_returned)} />
            <LogLine k="top_score" v={String(d.top_score ?? "n/a")} />
            {(d.failures as string[]).map((f, i) => (
              <div key={i} className="text-red-400">✗ {f}</div>
            ))}
            <LogLine k="next" v={String(d.recommendation)} />
          </>)}

          {isRetry && (<>
            <LogLine k="trigger"  v={String(d.reason)} />
            <LogLine k="filter"   v={String(d.filter_loosened)} />
            <LogLine k="new_query" v={String(d.new_query)} color="text-amber-300" />
            <LogLine k="sections" v={(d.new_sections as string[]).join(", ") || "all"} />
          </>)}

          {step.step === "generate" && (<>
            <LogLine k="chunks_in"   v={String(d.chunks_used)} />
            <LogLine k="citations"   v={String(d.citations_used)} />
            <LogLine k="elapsed"     v={`${d.elapsed_sec}s`} />
          </>)}
        </div>
      )}
    </div>
  );
}

function LogLine({ k, v, color }: { k: string; v: string; color?: string }) {
  return (
    <div className="flex gap-2 leading-relaxed">
      <span className="text-slate-600 min-w-20">{k}</span>
      <span className={color ?? "text-slate-300"}>{v}</span>
    </div>
  );
}

// ── Main ───────────────────────────────────────────────
export default function Home() {
  const [query,   setQuery]   = useState("");
  const [result,  setResult]  = useState<QueryResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<string | null>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (result) resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [result]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim() || loading) return;
    setLoading(true); setError(null); setResult(null);
    try {
      const res  = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query.trim() }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Request failed");
      setResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  const conf = result?.confidence;
  const confidencePct = Math.round((conf?.confidence ?? 0) * 100);

  return (
    <main className="min-h-screen bg-[#0A0F1E] text-white">

      {/* ── Ticker tape ── */}
      <TickerTape />

      {/* ── Header ── */}
      <header className="border-b border-slate-800 bg-[#0A0F1E]/95 backdrop-blur sticky top-0 z-20">
        <div className="max-w-7xl mx-auto px-6 py-5 flex items-start justify-between gap-6">
          <div className="space-y-1">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-amber-500 rounded flex items-center justify-center flex-shrink-0">
                <TrendingUp className="w-4 h-4 text-black" />
              </div>
              <h1 className="text-xl font-bold tracking-tight">
                SEC Filing Intelligence
              </h1>
            </div>
            <p className="text-slate-400 text-sm pl-11">
              Ask a question. Watch the agent route, rewrite, and self-correct. Get a cited answer from real 10-K filings.
            </p>
          </div>
          <div className="hidden md:flex items-center gap-4 text-xs text-slate-500 flex-shrink-0 pt-1">
            <span className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              10 companies indexed
            </span>
            <span className="flex items-center gap-1.5">
              <Terminal className="w-3 h-3" />
              Agentic RAG
            </span>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-10 space-y-10">

        {/* ── Search ── */}
        <div className="max-w-3xl mx-auto space-y-4">
          <form onSubmit={handleSubmit} className="space-y-3">
            {/* Terminal-style input */}
            <div className="relative flex items-start gap-3 bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 focus-within:border-amber-500/60 transition-colors">
              <span className="font-mono text-amber-500 text-sm mt-0.5 flex-shrink-0 select-none">›</span>
              <textarea
                value={query}
                onChange={e => setQuery(e.target.value)}
                onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(e); }}}
                placeholder="Ask about any company's filings…"
                rows={2}
                maxLength={500}
                className="flex-1 bg-transparent text-slate-100 placeholder-slate-600 text-sm focus:outline-none resize-none font-mono"
              />
            </div>
            <div className="flex items-center justify-between">
              <p className="text-xs text-slate-600 font-mono">{query.length}/500 chars · shift+enter for newline</p>
              <button
                type="submit"
                disabled={loading || !query.trim()}
                className="inline-flex items-center gap-2 rounded-lg bg-amber-500 hover:bg-amber-400 text-black px-5 py-2 text-sm font-bold disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                {loading
                  ? <><Loader2 className="w-4 h-4 animate-spin" /> Running agent…</>
                  : <><Zap className="w-4 h-4" /> Run agent</>}
              </button>
            </div>
          </form>

          {/* Sample queries */}
          {!result && !loading && (
            <div className="flex flex-wrap gap-2">
              {SAMPLE_QUERIES.map(q => (
                <button
                  key={q}
                  onClick={() => setQuery(q)}
                  className="text-xs px-3 py-1.5 rounded-full border border-slate-700 text-slate-400 hover:border-amber-500/50 hover:text-amber-400 transition-colors font-mono"
                >
                  {q}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* ── Error ── */}
        {error && (
          <div className="max-w-3xl mx-auto rounded-lg bg-red-500/10 border border-red-500/30 px-4 py-3 text-sm text-red-400 font-mono">
            ✗ {error}
          </div>
        )}

        {/* ── Loading ── */}
        {loading && (
          <div className="max-w-3xl mx-auto space-y-2 animate-pulse">
            {["w-1/3", "w-full", "w-5/6", "w-2/3"].map((w, i) => (
              <div key={i} className={`h-3 bg-slate-800 rounded ${w}`} />
            ))}
          </div>
        )}

        {/* ── Results ── */}
        {result && (
          <div ref={resultRef} className="space-y-6">

            {/* Stats bar */}
            <div className="flex flex-wrap items-center gap-3 pb-4 border-b border-slate-800">
              <div className="flex items-center gap-4 text-xs font-mono">
                <span className="text-slate-500">query</span>
                <span className="text-slate-200 italic">"{result.query}"</span>
              </div>
              <div className="flex items-center gap-3 ml-auto">
                {[
                  { label: "attempts", value: String(result.attempts), color: result.attempts > 1 ? "text-amber-400" : "text-emerald-400" },
                  { label: "confidence", value: `${confidencePct}%`, color: confidencePct >= 75 ? "text-emerald-400" : confidencePct >= 50 ? "text-amber-400" : "text-red-400" },
                  { label: "time", value: `${result.elapsed_sec}s`, color: "text-slate-300" },
                ].map(s => (
                  <div key={s.label} className="text-center">
                    <p className={`text-base font-bold font-mono ${s.color}`}>{s.value}</p>
                    <p className="text-xs text-slate-600">{s.label}</p>
                  </div>
                ))}
                {result.attempts > 1 && (
                  <span className="flex items-center gap-1.5 text-xs font-mono text-amber-400 bg-amber-500/10 border border-amber-500/20 px-3 py-1 rounded-full">
                    <RefreshCw className="w-3 h-3" />
                    self-corrected {result.attempts - 1}×
                  </span>
                )}
              </div>
            </div>

            {/* Main layout */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

              {/* ── Agent trace — left ── */}
              <div className="lg:col-span-4 xl:col-span-3">
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 sticky top-24">
                  <div className="flex items-center gap-2 mb-4 pb-3 border-b border-slate-800">
                    <Terminal className="w-4 h-4 text-amber-500" />
                    <span className="text-xs font-mono font-bold text-slate-300 uppercase tracking-widest">
                      Agent process log
                    </span>
                  </div>
                  <div className="space-y-3">
                    {result.trace.map((step, i) => (
                      <TraceStep key={i} step={step} index={i} />
                    ))}
                  </div>
                </div>
              </div>

              {/* ── Answer + sources — right ── */}
              <div className="lg:col-span-8 xl:col-span-9 space-y-5">

                {/* Answer */}
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-5">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <GitBranch className="w-4 h-4 text-amber-500" />
                      <span className="text-sm font-semibold text-slate-200">Answer</span>
                    </div>
                    <div className={`flex items-center gap-1.5 text-xs font-mono font-semibold ${
                      conf?.is_sufficient ? "text-emerald-400" : "text-amber-400"
                    }`}>
                      {conf?.is_sufficient
                        ? <CheckCircle className="w-3.5 h-3.5" />
                        : <AlertCircle className="w-3.5 h-3.5" />}
                      {confidencePct}% confidence
                    </div>
                  </div>

                  <AnswerText text={result.answer} />

                  {/* Citations */}
                  {result.cited_chunks.length > 0 && (
                    <div className="pt-4 border-t border-slate-800 space-y-3">
                      <p className="text-xs font-mono text-slate-600 uppercase tracking-widest">
                        Sources cited
                      </p>
                      {result.cited_chunks.map(c => (
                        <div key={c.citation_number} className="flex gap-3">
                          <span className="inline-flex items-center justify-center w-5 h-5 bg-amber-500 text-black text-xs font-bold rounded font-mono flex-shrink-0 mt-0.5">
                            {c.citation_number}
                          </span>
                          <div className="space-y-1">
                            <div className="flex items-center gap-2 flex-wrap">
                              <span className="text-xs font-bold text-slate-300 font-mono">{c.ticker}</span>
                              <span className="text-xs text-slate-500">{c.company}</span>
                              <SectionBadge section={c.section} />
                            </div>
                            <p className="text-xs text-slate-500 leading-relaxed">
                              {c.text_snippet.replace(/&#\d+;|&[a-z]+;/g, " ").substring(0, 160)}…
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Sources */}
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
                  <div className="flex items-center gap-2">
                    <Search className="w-4 h-4 text-slate-500" />
                    <span className="text-xs font-mono font-bold text-slate-400 uppercase tracking-widest">
                      Retrieved chunks
                    </span>
                    <span className="text-xs text-slate-600 ml-auto font-mono">
                      top {result.sources.length} after reranking
                    </span>
                  </div>

                  <div className="space-y-2 max-h-72 overflow-y-auto">
                    {result.sources.map((s, i) => (
                      <div
                        key={i}
                        className="flex gap-3 p-3 rounded-lg border border-slate-800 hover:border-amber-500/30 transition-colors"
                      >
                        <span className="font-mono text-xs font-bold text-amber-500 mt-0.5 flex-shrink-0 min-w-4">
                          {i + 1}
                        </span>
                        <div className="space-y-1.5 min-w-0">
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="font-mono text-xs font-bold text-slate-300">{s.ticker}</span>
                            <SectionBadge section={s.section} />
                            <span className="ml-auto font-mono text-xs text-slate-600">
                              {s.rerank_score?.toFixed(2)}
                            </span>
                          </div>
                          <p className="text-xs text-slate-500 leading-relaxed line-clamp-2">
                            {s.text.replace(/&#\d+;|&[a-z]+;/g, " ").substring(0, 200)}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}