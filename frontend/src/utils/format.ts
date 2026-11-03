export const percent = (value: number | null | undefined, digits = 1) =>
  value == null || Number.isNaN(value) ? '—' : (value * 100).toFixed(digits) + '%'

export const decimal = (value: number | null | undefined, digits = 3) =>
  value == null || Number.isNaN(value) ? '—' : value.toFixed(digits)

export const shortId = (value: string | null | undefined, length = 12) => {
  if (!value) return '—'
  return value.length > length ? value.slice(0, length) + '…' : value
}

export const dateTime = (value: string | null | undefined) => {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(date)
}

export const words = (value: string) =>
  value.replaceAll('_', ' ').toLowerCase().replace(/^./, (char) => char.toUpperCase())
