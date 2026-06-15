// lib/api.ts
// ----------
// Typed API client for the AYU backend.
// All fetch calls live here — components never call fetch directly.
// This makes it trivial to swap the base URL or add auth headers later.

export const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

// ---------------------------------------------------------------------------
// Types — mirror the backend Pydantic schemas
// ---------------------------------------------------------------------------

export type MetricStatus = "normal" | "high" | "low" | "borderline" | "unknown";

export interface Biomarker {
  name: string;
  value: number | null;
  unit: string | null;
  reference_range: string | null;
  status: MetricStatus;
  plain_explanation: string | null;
}

export interface ExtractedReport {
  report_type: string;
  patient_name: string | null;
  report_date: string | null;
  lab_name: string | null;
  biomarkers: Biomarker[];
  abnormal_flags: string[];
  health_summary: string;
  educational_notes: string[];
  extraction_confidence: number;
}

export interface AnalysisResponse {
  success: boolean;
  filename: string;
  page_count: number;
  char_count: number;
  report: ExtractedReport | null;
  error: string | null;
  disclaimer: string;
}
export interface ChatSource {
  source: string;
  topic: string;
  chunk_index: number;
  similarity_score: number;
}

export interface ChatResponse {
  success: boolean;
  answer: string;
  sources: ChatSource[];
  confidence: string;
  disclaimer: string;
}
// ---------------------------------------------------------------------------
// API calls
// ---------------------------------------------------------------------------

export async function analyzeReport(file: File): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/reports/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    // Handle HTTP errors (4xx, 5xx) that aren't caught by our success=false pattern
    let errorMessage = `Server error: ${response.status}`;
    try {
      const errorData = await response.json();
      errorMessage = errorData.detail || errorData.error || errorMessage;
    } catch {
      // Response wasn't JSON
    }
    throw new Error(errorMessage);
  }

  return response.json() as Promise<AnalysisResponse>;
}

export async function askAyu(question: string): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE}/chat/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
    }),
  });

  if (!response.ok) {
    let errorMessage = `Server error: ${response.status}`;

    try {
      const errorData = await response.json();
      errorMessage =
        errorData.detail?.error ||
        errorData.detail ||
        errorData.error ||
        errorMessage;
    } catch {
      // ignore
    }

    throw new Error(errorMessage);
  }

  return response.json() as Promise<ChatResponse>;
}