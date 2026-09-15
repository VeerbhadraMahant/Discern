import type { AnalyzeResponse } from "./types";

// In dev, Vite proxies /api to http://localhost:8000 (see vite.config.ts), so
// API_BASE stays empty. In production (e.g. Vercel), there is no such proxy -
// set VITE_API_URL to the deployed backend's origin (no trailing slash).
const API_BASE = import.meta.env.VITE_API_URL ?? "";

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

// Fetch a remote image (e.g. a sample scenario photo) and convert it into a
// File so it can be sent through the same real analyzeFrame() pipeline as a
// user upload, instead of relying on any pre-written mock result.
export async function urlToFile(url: string, filename: string): Promise<File> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch sample image (${res.status})`);
  const blob = await res.blob();
  return new File([blob], filename, { type: blob.type || "image/jpeg" });
}
