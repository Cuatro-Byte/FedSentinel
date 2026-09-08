import type { ReactNode } from 'react'
import { Activity, Database, LoaderCircle } from 'lucide-react'

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
    <section className={'panel ' + className} id={id}>
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

export function StatusBadge({ value }: { value: string | null | undefined }) {
  const status = value || 'UNKNOWN'
  return <span className={'status-badge status-' + status.toLowerCase()}>{status.replaceAll('_', ' ')}</span>
}

export function EmptyState({
  title,
  description,
}: {
  title: string
  description: string
}) {
  return (
    <div className="empty-state">
      <Database aria-hidden="true" />
      <strong>{title}</strong>
      <span>{description}</span>
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
  tone?: 'neutral' | 'accent' | 'warning' | 'success'
  icon?: ReactNode
}) {
  return (
    <article className={'metric-card metric-' + tone}>
      <div className="metric-card-top">
        <span>{label}</span>
        <span className="metric-icon">{icon}</span>
      </div>
      <strong className="metric-value">{value}</strong>
      <span className="metric-context">{context}</span>
    </article>
  )
}
