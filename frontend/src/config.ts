export const DEFAULT_URL = "http://127.0.0.1:7860";

export interface AppConfig {
  url: string;
  mock: boolean;
  /** Set when ?space= was present but was not an allowed Space or local URL, so it was ignored. */
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

const LOCAL_HOSTS = ["localhost", "127.0.0.1", "[::1]"];

/**
 * A link may only point the app at a Hugging Face Space (https, *.hf.space), a local server, or the
 * build-time URL. Otherwise a crafted `?space=` link would send the visitor's uploads to a third party.
 */
export function isTrustedSpace(value: string, envUrl: string): boolean {
  try {
    const u = new URL(value);
    if (u.username || u.password) return false;
    if (u.origin === new URL(envUrl).origin) return true;
    if (u.protocol === "http:" && LOCAL_HOSTS.includes(u.hostname)) return true;
    return u.protocol === "https:" && u.hostname.endsWith(".hf.space") && u.hostname.length > ".hf.space".length;
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
    if (isHttpUrl(space) && isTrustedSpace(space, envUrl)) url = space;
    else badSpace = space;
  }
  const mockParam = params.get("mock");
  const mock = mockParam !== null ? mockParam === "1" : env.VITE_USE_MOCK === "1";
  return { url, mock, badSpace };
}
