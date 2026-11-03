import { Activity, Crosshair, ShieldCheck, TrendingUp } from 'lucide-react'
import type { MetricRecord } from '../../types'
import { decimal, percent } from '../../utils/format'
import { EmptyState, MetricCard, Panel } from '../shared/Ui'
import { LineChart } from '../shared/LineChart'

export function ModelMetrics({ metrics }: { metrics: MetricRecord[] }) {
  const latest = metrics.at(-1)
  return (
    <section id="metrics" className="metrics-section">
      <div className="metric-grid">
        <MetricCard label="Model accuracy" value={percent(latest?.accuracy)} context={latest ? 'Backend evaluation · round ' + latest.round_id : 'Awaiting evaluation'} tone="success" icon={<TrendingUp aria-hidden="true" />} />
        <MetricCard label="Model loss" value={decimal(latest?.loss)} context="Persisted evaluation output" icon={<Activity aria-hidden="true" />} />
        <MetricCard label="Attack success" value={percent(latest?.attack_success_rate)} context={latest?.attack_success_rate == null ? 'Not applicable this round' : 'Accepted attacked updates'} tone="warning" icon={<Crosshair aria-hidden="true" />} />
        <MetricCard label="Recovery events" value={String(metrics.reduce((sum, item) => sum + item.recovery_count, 0))} context="Across the selected run" tone="accent" icon={<ShieldCheck aria-hidden="true" />} />
      </div>
      <Panel title="Model health over rounds" eyebrow="Real evaluation metrics" className="model-chart">
        {!metrics.length ? <EmptyState title="No model metrics" description="Accuracy and loss are rendered only from persisted backend results." /> : (
          <>
            <div className="chart-legend"><span><i className="legend-accuracy" />Accuracy</span><span><i className="legend-loss" />Loss</span></div>
            <LineChart labels={metrics.map((item) => item.round_id)} series={[{ label: 'Accuracy', values: metrics.map((item) => item.accuracy), className: 'line-accuracy' }, { label: 'Loss', values: metrics.map((item) => item.loss), className: 'line-loss' }]} valueFormatter={(value) => value.toFixed(2)} ariaLabel="Model accuracy and loss by federated round" />
          </>
        )}
      </Panel>
    </section>
  )
}
