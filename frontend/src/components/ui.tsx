import { CircleNotch, WarningCircle } from "@phosphor-icons/react";
import type { ButtonHTMLAttributes, ReactNode } from "react";
import type { DiscernError } from "../api/errors";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "link";
  loading?: boolean;
  loadingLabel?: string;
  icon?: ReactNode;
}

export function Button({
  variant = "primary",
  loading = false,
  loadingLabel = "Working",
  icon,
  disabled,
  className = "",
  children,
  type = "button",
  ...rest
}: ButtonProps) {
  const off = disabled || loading;
  const base =
    variant === "primary"
      ? "inline-flex min-h-11 items-center justify-center gap-2 rounded-tag bg-ink px-5 text-parchment"
      : "link";
  return (
    <button
      type={type}
      disabled={off}
      aria-disabled={off || undefined}
      className={`${base} ${off ? "cursor-not-allowed opacity-45" : ""} ${className}`}
      {...rest}
    >
      {loading ? <CircleNotch size={20} className="spin" aria-hidden="true" /> : icon}
      <span>{loading ? loadingLabel : children}</span>
    </button>
  );
}

export function Card({
  children,
  className = "",
  as: Tag = "div",
  ...rest
}: { children: ReactNode; className?: string; as?: "div" | "section" | "li" | "article" } & React.HTMLAttributes<HTMLElement>) {
  return (
    <Tag className={`rounded-card bg-bone p-6 text-ink shadow-card ${className}`} {...rest}>
      {children}
    </Tag>
  );
}

/** Status stamp. Outlined in ember with ink text, because ember text on parchment is below 4.5:1. */
export function Stamp({ children, icon }: { children: ReactNode; icon?: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1 whitespace-nowrap rounded-tag border-2 border-ember bg-parchment px-2 py-0.5 type-label text-ink">
      {icon}
      {children}
    </span>
  );
}

export function Field({
  id,
  label,
  helper,
  error,
  children,
}: {
  id: string;
  label: string;
  helper?: string;
  error?: string | null;
  children: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor={id} className="font-semibold">
        {label}
      </label>
      {children}
      {helper && (
        <p id={`${id}-help`} className="type-body-s">
          {helper}
        </p>
      )}
      {error && (
        <p id={`${id}-err`} role="alert" className="flex items-start gap-1 type-label">
          <WarningCircle size={20} className="shrink-0 text-ember" aria-hidden="true" />
          <span>{error}</span>
        </p>
      )}
    </div>
  );
}

export const inputClass =
  "min-h-11 w-full rounded-tag border border-ink bg-bone px-3 py-2 text-ink placeholder:text-ink/70";

export function ErrorNotice({ error, onRetry, onDismiss }: { error: DiscernError; onRetry?: () => void; onDismiss?: () => void }) {
  return (
    <div role="alert" className="flex flex-col gap-2 rounded-tag border-2 border-ember bg-parchment p-4">
      <p className="flex items-start gap-2 font-semibold">
        <WarningCircle size={24} className="shrink-0 text-ember" aria-hidden="true" />
        <span>{error.message}</span>
      </p>
      <p className="font-normal">{error.hint}</p>
      <div className="flex flex-wrap gap-4">
        {error.retryable && onRetry && <Button onClick={onRetry}>Retry</Button>}
        {onDismiss && (
          <Button variant="link" onClick={onDismiss}>
            Dismiss
          </Button>
        )}
      </div>
    </div>
  );
}

export function ProgressBar({ value, label }: { value: number | null; label: string }) {
  const pct = value === null ? null : Math.round(Math.min(1, Math.max(0, value)) * 100);
  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct ?? undefined}
      className="h-3 w-full border border-ink bg-parchment"
    >
      <div className={`h-full bg-ink ${pct === null ? "w-1/3 animate-pulse" : ""}`} style={pct === null ? undefined : { width: `${pct}%` }} />
    </div>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`skeleton animate-pulse ${className}`} aria-hidden="true" />;
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <Card className="flex flex-col gap-2">
      <h2 className="type-h2">{title}</h2>
      {children}
    </Card>
  );
}
