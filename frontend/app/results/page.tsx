"use client";

/**
 * page.tsx — AYU Prototype Main Page
 *
 * Single-page application that handles:
 *   1. PDF upload with drag-and-drop
 *   2. Loading / progress state during analysis
 *   3. Results display (summary, biomarkers grid, educational notes)
 *   4. Error handling
 *
 * Design language:
 *   Deep teal (#0F4C5C) as the primary grounding color — clinical but calm.
 *   Warm off-white (#F7F4EF) background — not the default white, not the trendy cream.
 *   Status colors are intentional: amber for borderline, rose for high/low.
 *   Type: system font stack with careful weight and tracking choices.
 *   Signature element: the vitals strip — a horizontal band showing abnormal markers
 *   before the user reads the summary, giving an at-a-glance reading like a monitor.
 */

import { useState, useCallback, useRef } from "react";
import { analyzeReport, askAyu, AnalysisResponse, Biomarker, ChatSource, MetricStatus } from "../../lib/api";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function statusColor(status: MetricStatus): string {
  switch (status) {
    case "high":
      return "bg-rose-50 border-rose-300 text-rose-800";
    case "low":
      return "bg-amber-50 border-amber-300 text-amber-800";
    case "borderline":
      return "bg-yellow-50 border-yellow-300 text-yellow-700";
    case "normal":
      return "bg-teal-50 border-teal-200 text-teal-800";
    default:
      return "bg-slate-50 border-slate-200 text-slate-600";
  }
}

function statusBadge(status: MetricStatus): string {
  switch (status) {
    case "high":
      return "bg-rose-100 text-rose-700";
    case "low":
      return "bg-amber-100 text-amber-700";
    case "borderline":
      return "bg-yellow-100 text-yellow-700";
    case "normal":
      return "bg-teal-100 text-teal-700";
    default:
      return "bg-slate-100 text-slate-500";
  }
}

function confidenceLabel(score: number): { label: string; color: string } {
  if (score >= 0.8) return { label: "High confidence", color: "text-teal-600" };
  if (score >= 0.5) return { label: "Moderate confidence", color: "text-amber-600" };
  return { label: "Low confidence — review manually", color: "text-rose-600" };
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function DisclaimerBanner() {
  return (
    <div className="w-full bg-amber-50 border border-amber-200 rounded-lg px-4 py-3 flex items-start gap-3 text-sm text-amber-800">
      <span className="text-lg leading-none mt-0.5">⚠️</span>
      <p>
        <strong>Educational use only.</strong> AYU analyses your report to help you understand it —
        it does not diagnose, prescribe, or replace medical advice.
        Always consult a qualified healthcare professional.
      </p>
    </div>
  );
}

function BiomarkerCard({ bm }: { bm: Biomarker }) {
  return (
    <div className={`rounded-xl border p-4 flex flex-col gap-2 ${statusColor(bm.status)}`}>
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-sm tracking-wide">{bm.name}</h3>
        <span
          className={`text-xs font-medium px-2 py-0.5 rounded-full uppercase tracking-wider ${statusBadge(bm.status)}`}
        >
          {bm.status}
        </span>
      </div>
      {bm.value !== null && (
        <div className="flex items-baseline gap-1.5">
          <span className="text-2xl font-bold tabular-nums">{bm.value}</span>
          {bm.unit && <span className="text-xs opacity-70">{bm.unit}</span>}
        </div>
      )}
      {bm.reference_range && <p className="text-xs opacity-60">Ref: {bm.reference_range}</p>}
      {bm.plain_explanation && (
        <p className="text-xs leading-relaxed opacity-80 border-t border-current border-opacity-10 pt-2 mt-1">
          {bm.plain_explanation}
        </p>
      )}
    </div>
  );
}

function VitalsStrip({ flags }: { flags: string[] }) {
  if (!flags.length) return null;
  return (
    <div className="w-full bg-rose-700 rounded-xl px-5 py-4">
      <p className="text-rose-200 text-xs font-semibold uppercase tracking-widest mb-2">Flagged markers</p>
      <div className="flex flex-wrap gap-2">
        {flags.map((f) => (
          <span
            key={f}
            className="bg-rose-900 bg-opacity-60 text-rose-100 text-sm font-medium px-3 py-1 rounded-full"
          >
            {f}
          </span>
        ))}
      </div>
    </div>
  );
}

function UploadZone({
  onFile,
  loading,
}: {
  onFile: (f: File) => void;
  loading: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) onFile(file);
    },
    [onFile]
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => !loading && inputRef.current?.click()}
      className={`
        w-full border-2 border-dashed rounded-2xl p-12 flex flex-col items-center gap-4 
        cursor-pointer transition-all duration-200 select-none
        ${dragging ? "border-teal-500 bg-teal-50" : "border-slate-300 bg-white hover:border-teal-400 hover:bg-teal-50/40"}
        ${loading ? "opacity-50 cursor-not-allowed" : ""}
      `}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,application/pdf"
        className="hidden"
        onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
        disabled={loading}
      />
      <div className="w-16 h-16 rounded-2xl bg-teal-700 flex items-center justify-center">
        <svg
          width="32"
          height="32"
          viewBox="0 0 24 24"
          fill="none"
          stroke="white"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14,2 14,8 20,8" />
          <line x1="12" y1="18" x2="12" y2="12" />
          <line x1="9" y1="15" x2="15" y2="15" />
        </svg>
      </div>
      <div className="text-center">
        <p className="font-semibold text-slate-700">{loading ? "Analysing your report…" : "Upload your medical report"}</p>
        <p className="text-sm text-slate-500 mt-1">
          {loading ? "AI is extracting your health metrics" : "Drag & drop or click · PDF up to 10 MB"}
        </p>
      </div>
      {!loading && (
        <button className="mt-2 bg-teal-700 hover:bg-teal-800 text-white text-sm font-semibold px-6 py-2.5 rounded-xl transition-colors">
          Choose PDF
        </button>
      )}
      {loading && (
        <div className="flex gap-1.5">
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              className="w-2 h-2 bg-teal-600 rounded-full animate-bounce"
              style={{ animationDelay: `${i * 0.15}s` }}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function ResultsPanel({ result }: { result: AnalysisResponse }) {
  const r = result.report!;
  const conf = confidenceLabel(r.extraction_confidence);

  const abnormalBiomarkers = r.biomarkers.filter(
    (b) => b.status === "high" || b.status === "low" || b.status === "borderline"
  );
  const normalBiomarkers = r.biomarkers.filter((b) => b.status === "normal");

  return (
    <div className="w-full flex flex-col gap-6">
      {/* Meta bar */}
      <div className="flex flex-wrap items-center gap-3 text-sm text-slate-500">
        <span className="bg-slate-100 px-3 py-1 rounded-full font-medium text-slate-700">{result.filename}</span>
        <span>
          {result.page_count} page{result.page_count !== 1 ? "s" : ""}
        </span>
        <span>·</span>
        <span>{r.report_type.replace(/_/g, " ")}</span>
        {r.lab_name && (
          <>
            <span>·</span>
            <span>{r.lab_name}</span>
          </>
        )}
        {r.report_date && (
          <>
            <span>·</span>
            <span>{r.report_date}</span>
          </>
        )}
        <span className={`ml-auto font-medium ${conf.color}`}
        >
          {conf.label} ({Math.round(r.extraction_confidence * 100)}%)
        </span>
      </div>

      {/* Vitals strip — signature element */}
      <VitalsStrip flags={r.abnormal_flags} />

      {/* Health summary */}
      <div className="bg-white rounded-2xl border border-slate-200 p-6">
        <h2 className="text-xs font-semibold uppercase tracking-widest text-teal-700 mb-3">Health summary</h2>
        <p className="text-slate-700 leading-relaxed text-[15px]">{r.health_summary}</p>
      </div>

      {/* Abnormal / borderline markers */}
      {abnormalBiomarkers.length > 0 && (
        <section>
          <h2 className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-3">Values requiring attention</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {abnormalBiomarkers.map((bm) => (
              <BiomarkerCard key={bm.name} bm={bm} />
            ))}
          </div>
        </section>
      )}

      {/* Normal markers */}
      {normalBiomarkers.length > 0 && (
        <section>
          <h2 className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-3">Normal values</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {normalBiomarkers.map((bm) => (
              <BiomarkerCard key={bm.name} bm={bm} />
            ))}
          </div>
        </section>
      )}

      {/* Educational notes */}
      {r.educational_notes.length > 0 && (
        <div className="bg-teal-900 rounded-2xl p-6 text-white">
          <h2 className="text-xs font-semibold uppercase tracking-widest text-teal-300 mb-4">Educational notes</h2>
          <ul className="flex flex-col gap-3">
            {r.educational_notes.map((note, i) => (
              <li key={i} className="flex items-start gap-3 text-sm text-teal-100 leading-relaxed">
                <span className="mt-0.5 text-teal-400 shrink-0">→</span>
                {note}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Disclaimer */}
      <DisclaimerBanner />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function Home() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Ask AYU state
  const [question, setQuestion] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatAnswer, setChatAnswer] = useState("");
  const [chatSources, setChatSources] = useState<ChatSource[]>([]);
  const [chatError, setChatError] = useState("");

  const handleFile = useCallback(async (file: File) => {
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const data = await analyzeReport(file);
      if (!data.success) {
        setError(data.error || "Analysis failed. Please try again.");
      } else {
        setResult(data);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Could not connect to the server. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }, []);

  const handleAskAyu = useCallback(async () => {
    const q = question.trim();
    if (!q) return;

    setChatLoading(true);
    setChatError("");
    setChatAnswer("");
    setChatSources([]);

    try {
      const response = await askAyu(q);
      setChatAnswer(response.answer);
      setChatSources(response.sources);
    } catch (err: unknown) {
      setChatError(err instanceof Error ? err.message : "Failed to get response from AYU");
    } finally {
      setChatLoading(false);
    }
  }, [question]);

  return (
    <main className="min-h-screen bg-[#F7F4EF] font-sans">
      {/* Header */}
      <header className="bg-[#0F4C5C] text-white">
        <div className="max-w-4xl mx-auto px-6 py-5 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 bg-teal-400 bg-opacity-30 rounded-xl flex items-center justify-center">
              <span className="font-bold text-sm tracking-tight">AYU</span>
            </div>
            <div>
              <p className="font-semibold text-[15px] leading-none">AYU Health</p>
              <p className="text-teal-300 text-xs mt-0.5">AI Health Intelligence · Prototype</p>
            </div>
          </div>
          {result && (
            <button
              onClick={() => {
                setResult(null);
                setError(null);
                setChatAnswer("");
                setChatSources([]);
                setChatError("");
                setQuestion("");
              }}
              className="text-teal-300 hover:text-white text-sm transition-colors"
            >
              ← New upload
            </button>
          )}
        </div>
      </header>

      {/* Body */}
      <div className="max-w-4xl mx-auto px-6 py-10">
        {!result && !loading && (
          <div className="flex flex-col gap-8">
            {/* Hero text */}
            <div className="text-center max-w-xl mx-auto">
              <h1 className="text-3xl font-bold text-slate-800 leading-tight tracking-tight">
                Understand your
                <br />
                <span className="text-[#0F4C5C]">medical report</span>
              </h1>
              <p className="text-slate-500 mt-3 text-[15px] leading-relaxed">
                Upload a blood test or lab report. AYU extracts your health metrics, explains each value in
                plain language, and highlights anything worth discussing with your doctor.
              </p>
            </div>

            <UploadZone onFile={handleFile} loading={loading} />
            <DisclaimerBanner />
          </div>
        )}

        {loading && (
          <div className="flex flex-col items-center gap-8">
            <UploadZone onFile={handleFile} loading={loading} />
            <div className="text-center text-slate-500 text-sm">
              <p>Extracting text from your PDF…</p>
              <p className="mt-1">Asking AI to identify your health markers…</p>
              <p className="mt-1 text-teal-600 font-medium">This usually takes 10–20 seconds.</p>
            </div>
          </div>
        )}

        {error && !loading && (
          <div className="flex flex-col gap-6">
            <div className="bg-rose-50 border border-rose-200 rounded-xl p-6 text-center">
              <p className="text-rose-700 font-semibold mb-2">Analysis failed</p>
              <p className="text-rose-600 text-sm">{error}</p>
              <button
                onClick={() => setError(null)}
                className="mt-4 bg-rose-700 hover:bg-rose-800 text-white text-sm font-semibold px-5 py-2 rounded-xl transition-colors"
              >
                Try again
              </button>
            </div>
          </div>
        )}

        {result && !loading && (
          <div className="flex flex-col gap-8">
            <ResultsPanel result={result} />

            <div className="bg-white border border-slate-200 rounded-2xl p-6">
              <h2 className="text-lg font-semibold text-slate-800 mb-4">Ask AYU</h2>

              <div className="flex gap-3">
                <input
                  type="text"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="What is HbA1c?"
                  className="flex-1 border border-slate-300 rounded-xl px-4 py-3"
                />

                <button
                  onClick={handleAskAyu}
                  disabled={chatLoading}
                  className="bg-teal-700 hover:bg-teal-800 text-white px-5 py-3 rounded-xl"
                >
                  {chatLoading ? "Thinking..." : "Ask"}
                </button>
              </div>

              {chatError && <div className="mt-4 text-rose-600 text-sm">{chatError}</div>}

              {chatAnswer && (
                <div className="mt-6">
                  <h3 className="font-semibold mb-2">Answer</h3>

                  <p className="text-slate-700 leading-relaxed">{chatAnswer}</p>

                  {chatSources.length > 0 && (
                    <div className="mt-4">
                      <h4 className="font-medium mb-2">Sources</h4>

                      <div className="flex flex-col gap-2">
                        {chatSources.map((source, index) => (
                          <div key={index} className="bg-slate-50 rounded-lg px-3 py-2 text-sm">
                            <strong>{source.source}</strong> {" • "}
                            {source.topic} {" • "}
                            score: {source.similarity_score?.toFixed(2)}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </main>
  );
}

