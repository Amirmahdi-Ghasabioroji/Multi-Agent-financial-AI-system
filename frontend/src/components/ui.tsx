import clsx from "clsx";
import {
  AlertCircle,
  Inbox,
  LoaderCircle,
  Radio,
  type LucideIcon,
} from "lucide-react";
import type { HTMLAttributes, ReactNode } from "react";

export function Panel({
  children,
  className,
  ...props
}: HTMLAttributes<HTMLDivElement>) {
  return (
    <section className={clsx("panel", className)} {...props}>
      {children}
    </section>
  );
}

export function SectionHeading({
  eyebrow,
  title,
  detail,
  action,
}: {
  eyebrow?: string;
  title: string;
  detail?: string;
  action?: ReactNode;
}) {
  return (
    <div className="section-heading">
      <div>
        {eyebrow && <p className="eyebrow">{eyebrow}</p>}
        <h2>{title}</h2>
        {detail && <p className="muted mt-1">{detail}</p>}
      </div>
      {action}
    </div>
  );
}

export function Badge({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "teal" | "amber" | "red" | "blue";
}) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}

export function StatusBadge({ status = "unknown" }: { status?: string }) {
  const normalized = status.toLowerCase();
  const tone =
    ["completed", "succeeded", "healthy", "ready", "online", "success"].includes(
      normalized,
    )
      ? "teal"
      : ["failed", "error", "offline", "cancelled"].includes(normalized)
        ? "red"
        : ["running", "queued", "pending"].includes(normalized)
          ? "amber"
          : "neutral";
  return <Badge tone={tone}>{status}</Badge>;
}

export function ModeBadge({ demo }: { demo?: boolean }) {
  return (
    <Badge tone={demo === false ? "teal" : "amber"}>
      <Radio size={11} aria-hidden />
      {demo === false ? "Live data" : "Demo / frozen data"}
    </Badge>
  );
}

export function Metric({
  label,
  value,
  detail,
  icon: Icon,
  tone = "neutral",
}: {
  label: string;
  value: ReactNode;
  detail?: ReactNode;
  icon?: LucideIcon;
  tone?: "neutral" | "teal" | "amber" | "red";
}) {
  return (
    <div className={`metric metric-${tone}`}>
      <div className="metric-label">
        {Icon && <Icon size={15} aria-hidden />}
        {label}
      </div>
      <div className="metric-value">{value}</div>
      {detail && <div className="metric-detail">{detail}</div>}
    </div>
  );
}

export function LoadingState({ label = "Loading data…" }: { label?: string }) {
  return (
    <div className="state-box" role="status">
      <LoaderCircle className="animate-spin" size={22} aria-hidden />
      <span>{label}</span>
    </div>
  );
}

export function EmptyState({
  title,
  detail,
  action,
}: {
  title: string;
  detail: string;
  action?: ReactNode;
}) {
  return (
    <div className="state-box">
      <Inbox size={24} aria-hidden />
      <strong>{title}</strong>
      <p>{detail}</p>
      {action}
    </div>
  );
}

export function ErrorState({
  message,
  retry,
}: {
  message: string;
  retry?: () => void;
}) {
  return (
    <div className="error-box" role="alert">
      <AlertCircle size={18} aria-hidden />
      <div>
        <strong>Unable to load</strong>
        <p>{message}</p>
      </div>
      {retry && (
        <button className="button button-secondary button-small" onClick={retry}>
          Retry
        </button>
      )}
    </div>
  );
}

export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <div className="field">
      <span className="field-label">{label}</span>
      {children}
      {hint && <span className="field-hint">{hint}</span>}
    </div>
  );
}

export function PageHeader({
  eyebrow,
  title,
  description,
  children,
}: {
  eyebrow: string;
  title: string;
  description: string;
  children?: ReactNode;
}) {
  return (
    <header className="page-header">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {children && <div className="page-actions">{children}</div>}
    </header>
  );
}
