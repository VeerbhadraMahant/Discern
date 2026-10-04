export type DiscernErrorKind = "backend" | "network" | "invalid";

/** Typed error carrying a visitor-friendly message and a retry hint. */
export class DiscernError extends Error {
  readonly kind: DiscernErrorKind;
  readonly retryable: boolean;
  readonly hint: string;

  constructor(kind: DiscernErrorKind, message: string, retryable: boolean, hint: string) {
    super(message);
    this.name = "DiscernError";
    this.kind = kind;
    this.retryable = retryable;
    this.hint = hint;
  }
}

export function backendError(message: string): DiscernError {
  const text = message.trim() || "The Discern app could not complete that request.";
  return new DiscernError("backend", text, true, "Try again. If it keeps failing, start over with a new file.");
}

export function networkError(cause?: unknown): DiscernError {
  const detail = cause instanceof Error && cause.message ? ` (${cause.message})` : "";
  return new DiscernError(
    "network",
    `Could not reach the Discern app${detail}.`,
    true,
    "Check your connection and that the app is running, then retry.",
  );
}

export function invalidError(what: string): DiscernError {
  return new DiscernError(
    "invalid",
    `The Discern app sent a reply this page does not understand (${what}).`,
    false,
    "The page and the app may be out of date with each other.",
  );
}

export function toDiscernError(e: unknown): DiscernError {
  return e instanceof DiscernError ? e : networkError(e);
}
