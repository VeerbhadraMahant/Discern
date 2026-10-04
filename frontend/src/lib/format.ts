/** The one timestamp format: m:ss under an hour, h:mm:ss from an hour up. Milliseconds appear only in the trace. */
export function formatTime(seconds: number): string {
  const total = Math.floor(Math.max(0, seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const ss = String(s).padStart(2, "0");
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
}

export function formatRange(start: number, end: number): string {
  return `${formatTime(start)} to ${formatTime(end)}`;
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatTtl(seconds: number): string {
  if (seconds % 3600 === 0 && seconds >= 3600) {
    const h = seconds / 3600;
    return `${h} ${h === 1 ? "hour" : "hours"}`;
  }
  if (seconds % 60 === 0 && seconds >= 60) {
    const m = seconds / 60;
    return `${m} ${m === 1 ? "minute" : "minutes"}`;
  }
  return `${seconds} seconds`;
}
