import type { ReactNode } from 'react'
import { Activity, AlertTriangle, Database, LoaderCircle, X } from 'lucide-react'

export function Panel({
  title,
  eyebrow,
  action,
  children,
  className = '',
  id,
}: {
  title: string
  eyebrow?: string
  action?: ReactNode
  children: ReactNode
  className?: string
  id?: string
}) {
  return (
    <section className={`panel ${className}`} id={id}>
      <header className="panel-header">
        <div>
          {eyebrow && <span className="eyebrow">{eyebrow}</span>}
          <h2>{title}</h2>
        </div>
        {action}
      </header>
      {children}
    </section>
  )
}

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow: string
  title: string
  description: string
  actions?: ReactNode
}) {
  return (
    <header className="page-header">
      <div>
        <span className="eyebrow">{eyebrow}</span>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </header>
  )
}

export function StatusBadge({ value }: { value: string | null | undefined }) {
  const status = value || 'UNKNOWN'
  const className = status.toLowerCase().replace(/[^a-z0-9]+/g, '-')
  return (
    <span className={`status-badge status-${className}`}>
      <i aria-hidden="true" />
      {status.replaceAll('_', ' ')}
    </span>
  )
}

export function EmptyState({
  title,
  description,
  icon,
}: {
  title: string
  description: string
  icon?: ReactNode
}) {
  return (
    <div className="empty-state">
      {icon || <Database aria-hidden="true" />}
      <strong>{title}</strong>
      <span>{description}</span>
    </div>
  )
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="empty-state error-state">
      <AlertTriangle aria-hidden="true" />
      <strong>Unable to load data</strong>
      <span>{message}</span>
      {onRetry && (
        <button type="button" className="secondary-button compact" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  )
}

export function LoadingState({ label = 'Reading backend analytics' }: { label?: string }) {
  return (
    <div className="loading-state" aria-live="polite">
      <LoaderCircle className="spin" aria-hidden="true" />
      <span>{label}</span>
    </div>
  )
}

export function Skeleton({ rows = 3 }: { rows?: number }) {
  return (
    <div className="skeleton" aria-label="Loading">
      {Array.from({ length: rows }, (_, index) => (
        <span key={index} />
      ))}
    </div>
  )
}

export function MetricCard({
  label,
  value,
  context,
  tone = 'neutral',
  icon = <Activity aria-hidden="true" />,
}: {
  label: string
  value: string
  context: string
  tone?: 'neutral' | 'accent' | 'warning' | 'success' | 'danger'
  icon?: ReactNode
}) {
  return (
    <article className={`metric-card metric-${tone}`}>
      <div className="metric-card-top">
        <span>{label}</span>
        <span className="metric-icon">{icon}</span>
      </div>
      <strong className="metric-value">{value}</strong>
      <span className="metric-context">{context}</span>
    </article>
  )
}

export function Drawer({
  open,
  title,
  subtitle,
  onClose,
  children,
}: {
  open: boolean
  title: string
  subtitle?: string
  onClose: () => void
  children: ReactNode
}) {
  if (!open) return null
  return (
    <div className="drawer-layer" role="presentation">
      <button type="button" className="drawer-backdrop" onClick={onClose} aria-label="Close details" />
      <aside className="drawer" role="dialog" aria-modal="true" aria-label={title}>
        <header>
          <div>
            <span className="eyebrow">{subtitle || 'Inspector'}</span>
            <h2>{title}</h2>
          </div>
          <button type="button" className="icon-button" onClick={onClose} aria-label="Close drawer">
            <X />
          </button>
        </header>
        <div className="drawer-body">{children}</div>
      </aside>
    </div>
  )
}

export function ComparisonCard({
  label,
  accuracy,
  loss,
  caption,
}: {
  label: string
  accuracy?: number | null
  loss?: number | null
  caption?: string
}) {
  const render = (value: number | null | undefined, percentValue = false) =>
    value == null ? '—' : percentValue ? `${(value * 100).toFixed(1)}%` : value.toFixed(3)
  return (
    <article className="comparison-card">
      <span>{label}</span>
      <div>
        <small>Accuracy</small>
        <strong>{render(accuracy, true)}</strong>
      </div>
      <div>
        <small>Loss</small>
        <strong>{render(loss)}</strong>
      </div>
      {caption && <p>{caption}</p>}
    </article>
  )
}

export function DefinitionGrid({ items }: { items: Array<{ label: string; value: ReactNode }> }) {
  return (
    <dl className="definition-grid">
      {items.map((item) => (
        <div key={item.label}>
          <dt>{item.label}</dt>
          <dd>{item.value}</dd>
        </div>
      ))}
    </dl>
  )
}
