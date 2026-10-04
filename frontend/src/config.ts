export const DEFAULT_URL = "http://127.0.0.1:7860";

export interface AppConfig {
  url: string;
  mock: boolean;
  /** Set when ?space= was present but not an http(s) URL. */
  badSpace: string | null;
}

export function isHttpUrl(value: string): boolean {
  try {
    const u = new URL(value);
    return u.protocol === "http:" || u.protocol === "https:";
  } catch {
    return false;
  }
}

/** Resolve the backend URL and mock flag from the query string and build-time env. */
export function resolveConfig(
  search: string = typeof location === "undefined" ? "" : location.search,
  env: Record<string, unknown> = import.meta.env,
): AppConfig {
  const params = new URLSearchParams(search);
  const space = params.get("space");
  const rawUrl = env.VITE_DISCERN_URL;
  const envUrl = typeof rawUrl === "string" && isHttpUrl(rawUrl) ? rawUrl : DEFAULT_URL;
  let url = envUrl;
  let badSpace: string | null = null;
  if (space) {
    if (isHttpUrl(space)) url = space;
    else badSpace = space;
  }
  const mockParam = params.get("mock");
  const mock = mockParam !== null ? mockParam === "1" : env.VITE_USE_MOCK === "1";
  return { url, mock, badSpace };
}
