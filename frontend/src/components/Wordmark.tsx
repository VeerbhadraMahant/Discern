/** The Discern mark: a square drawn as four corner ticks (the frame brackets) with one ember dot at its centre ("the thing found"). */
export function WordmarkMark({ size = 28, className = "" }: { size?: number; className?: string }) {
  return (
    <svg viewBox="0 0 32 32" width={size} height={size} className={className} aria-hidden="true" focusable="false">
      <path
        d="M3 11V3h8M21 3h8v8M29 21v8h-8M11 29H3v-8"
        fill="none"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="square"
        strokeLinejoin="miter"
      />
      <circle cx="16" cy="16" r="4.5" className="fill-ember" />
    </svg>
  );
}

/** Mark plus the word in the display face. Colour follows the surrounding text (ink on light, parchment on ink); the dot stays ember. */
export function Wordmark({ href, className = "" }: { href?: string; className?: string }) {
  const inner = (
    <>
      <WordmarkMark />
      <span className="type-wordmark">Discern</span>
    </>
  );
  const cls = `inline-flex min-h-11 items-center gap-2 ${className}`;
  return href ? (
    <a href={href} className={cls}>
      {inner}
    </a>
  ) : (
    <span className={cls}>{inner}</span>
  );
}
