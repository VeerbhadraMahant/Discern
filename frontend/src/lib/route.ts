import { useCallback, useEffect, useState } from "react";

export const ROUTES = ["clean", "ask", "trace", "feedback", "about"] as const;
export type Route = (typeof ROUTES)[number];

export function parseHash(hash: string): Route {
  const name = hash.replace(/^#\/?/, "").split(/[?/]/)[0] ?? "";
  return (ROUTES as readonly string[]).includes(name) ? (name as Route) : "clean";
}

export function useHashRoute(): [Route, (r: Route) => void] {
  const [route, setRoute] = useState<Route>(() => parseHash(location.hash));
  useEffect(() => {
    const onChange = () => setRoute(parseHash(location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  const navigate = useCallback((r: Route) => {
    location.hash = `#/${r}`;
  }, []);
  return [route, navigate];
}
