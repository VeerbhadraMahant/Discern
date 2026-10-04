import { useCallback, useEffect, useState } from "react";

export const ROUTES = ["clean", "ask", "trace", "feedback", "about"] as const;
/** The five app tabs. */
export type Tab = (typeof ROUTES)[number];
/** The landing page is the default route: an empty hash, `#/` or any hash that is not a tab name. */
export type Route = "home" | Tab;

export function parseHash(hash: string): Route {
  const name = hash.replace(/^#\/?/, "").split(/[?/]/)[0] ?? "";
  return (ROUTES as readonly string[]).includes(name) ? (name as Tab) : "home";
}

export function useHashRoute(): [Route, (r: Route) => void] {
  const [route, setRoute] = useState<Route>(() => parseHash(location.hash));
  useEffect(() => {
    const onChange = () => setRoute(parseHash(location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  const navigate = useCallback((r: Route) => {
    location.hash = r === "home" ? "#/" : `#/${r}`;
  }, []);
  return [route, navigate];
}
