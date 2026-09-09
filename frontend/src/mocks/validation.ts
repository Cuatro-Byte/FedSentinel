import type { ValidationMetrics } from '../types/navigation'

// Development-only contract preview. These values are never mixed with API data.
export const validationPreview: ValidationMetrics = {
  source: 'mock',
  detection: { tp: 17, fp: 2, tn: 74, fn: 3, precision: 0.895, recall: 0.85, f1: 0.872, falsePositiveRate: 0.026, detectionRate: 0.85 },
  threatScores: { benign: [0.08, 0.13, 0.18, 0.22, 0.29, 0.34], attacker: [0.56, 0.68, 0.79, 0.86, 0.92], threshold: 0.5 },
  impact: { predictedDegradation: 0.184, observedDegradation: 0.171, absoluteError: 0.013 },
  recovery: { inducedLoss: 0.171, recoveredPerformance: 0.142, gain: 0.142, effectiveness: 0.83 },
}
