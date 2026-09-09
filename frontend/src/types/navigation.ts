export type AppRoute =
  | '/dashboard'
  | '/simulation'
  | '/rounds'
  | '/clients'
  | '/threats'
  | '/impact'
  | '/recovery'
  | '/validation'
  | '/audit'

export interface ValidationMetrics {
  source: 'mock' | 'api'
  detection: {
    tp: number
    fp: number
    tn: number
    fn: number
    precision: number
    recall: number
    f1: number
    falsePositiveRate: number
    detectionRate: number
  }
  threatScores: { benign: number[]; attacker: number[]; threshold: number }
  impact: { predictedDegradation: number; observedDegradation: number; absoluteError: number }
  recovery: { inducedLoss: number; recoveredPerformance: number; gain: number; effectiveness: number }
}
