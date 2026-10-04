export function formatTime(seconds: number): string {
  const s = Math.max(0, seconds);
  const m = Math.floor(s / 60);
  const rest = s - m * 60;
  return `${String(m).padStart(2, "0")}:${rest.toFixed(1).padStart(4, "0")}`;
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
