import { useEffect, useState } from "react";

/** True when the visitor asked for reduced motion. Falls back to false where matchMedia is missing. */
export function prefersReducedMotion(): boolean {
  try {
    return typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch {
    return false;
  }
}

export function canObserve(): boolean {
  return typeof IntersectionObserver === "function";
}

/** Scroll-spy: the id of the section nearest the top band of the viewport, or null without IntersectionObserver. */
export function useScrollSpy(ids: readonly string[]): string | null {
  const [active, setActive] = useState<string | null>(null);
  useEffect(() => {
    if (!canObserve()) return;
    const seen = new Map<string, boolean>();
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) seen.set(e.target.id, e.isIntersecting);
        const first = ids.find((id) => seen.get(id));
        setActive(first ?? null);
      },
      { rootMargin: "-20% 0px -65% 0px" },
    );
    for (const id of ids) {
      const el = document.getElementById(id);
      if (el) io.observe(el);
    }
    return () => io.disconnect();
  }, [ids]);
  return active;
}

/** Scroll to a section and move focus there, without changing the hash route. */
export function goToSection(id: string): void {
  const el = document.getElementById(id);
  if (!el) return;
  const smooth = !prefersReducedMotion();
  el.scrollIntoView?.({ behavior: smooth ? "smooth" : "auto", block: "start" });
  el.focus({ preventScroll: true });
}
