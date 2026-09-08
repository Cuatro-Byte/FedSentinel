import { describe, expect, it } from 'vitest'
import { dateTime, decimal, percent, shortId, words } from './format'

describe('analytics formatting', () => {
  it('keeps missing backend values visibly missing', () => {
    expect(percent(null)).toBe('—')
    expect(decimal(undefined)).toBe('—')
  })

  it('formats real numeric values without changing them', () => {
    expect(percent(0.914)).toBe('91.4%')
    expect(decimal(0.12549)).toBe('0.125')
  })

  it('formats identifiers and labels', () => {
    expect(shortId('model-version-long', 8)).toBe('model-ve…')
    expect(words('DOWN_WEIGHT')).toBe('Down weight')
    expect(dateTime(null)).toBe('—')
  })
})
