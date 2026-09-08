interface Series {
  label: string
  values: number[]
  className: string
}

export function LineChart({
  labels,
  series,
  valueFormatter,
  ariaLabel,
}: {
  labels: number[]
  series: Series[]
  valueFormatter: (value: number) => string
  ariaLabel: string
}) {
  const width = 760
  const height = 250
  const padding = { top: 24, right: 20, bottom: 38, left: 48 }
  const allValues = series.flatMap((item) => item.values)
  const max = Math.max(...allValues, 1)
  const min = Math.min(...allValues, 0)
  const span = max - min || 1
  const x = (index: number) =>
    padding.left + (index / Math.max(labels.length - 1, 1)) * (width - padding.left - padding.right)
  const y = (value: number) =>
    padding.top + ((max - value) / span) * (height - padding.top - padding.bottom)
  const gridValues = [max, min + span * 0.5, min]

  return (
    <div className="chart-wrap">
      <svg viewBox={'0 0 ' + width + ' ' + height} role="img" aria-label={ariaLabel}>
        <title>{ariaLabel}</title>
        {gridValues.map((value) => (
          <g key={value}>
            <line className="chart-grid" x1={padding.left} x2={width - padding.right} y1={y(value)} y2={y(value)} />
            <text className="chart-label" x={padding.left - 10} y={y(value) + 4} textAnchor="end">
              {valueFormatter(value)}
            </text>
          </g>
        ))}
        {series.map((item) => {
          const points = item.values.map((value, index) => x(index) + ',' + y(value)).join(' ')
          return (
            <g key={item.label}>
              <polyline className={'chart-line ' + item.className} points={points} />
              {item.values.map((value, index) => (
                <circle key={index} className={'chart-dot ' + item.className} cx={x(index)} cy={y(value)} r="3.5">
                  <title>{item.label + ', round ' + labels[index] + ': ' + valueFormatter(value)}</title>
                </circle>
              ))}
            </g>
          )
        })}
        {labels.map((label, index) => (
          <text key={label} className="chart-label" x={x(index)} y={height - 12} textAnchor="middle">
            R{label}
          </text>
        ))}
      </svg>
    </div>
  )
}
