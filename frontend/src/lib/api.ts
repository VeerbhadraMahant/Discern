import type { AnalyzeResponse } from "./types";

const API_BASE = ""; // Vite proxies /api to http://localhost:8000 in dev

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(2000) });
    if (!res.ok) return false;
    const data = await res.json();
    return data.status === "ok";
  } catch {
    return false;
  }
}

export async function analyzeFrame(file: File): Promise<AnalyzeResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    let errorDetail = "Analysis request failed";
    try {
      const err = await res.json();
      errorDetail = err.detail || errorDetail;
    } catch {
      errorDetail = `Server error (${res.status} ${res.statusText})`;
    }
    throw new Error(errorDetail);
  }

  const data: AnalyzeResponse = await res.json();
  return data;
}

// Convert a base64 or SVG data URL into a File object for real backend analysis
export async function dataUrlToFile(dataUrl: string, filename: string): Promise<File> {
  const res = await fetch(dataUrl);
  const blob = await res.blob();
  return new File([blob], filename, { type: blob.type || "image/jpeg" });
}
