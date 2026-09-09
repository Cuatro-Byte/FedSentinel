import type { ValidationMetrics } from '../types/navigation'

export async function getValidationMetrics(): Promise<ValidationMetrics | null> {
  const usePreview = import.meta.env.DEV || import.meta.env.VITE_ENABLE_VALIDATION_MOCKS === 'true'
  if (!usePreview) return null
  const { validationPreview } = await import('../mocks/validation')
  return validationPreview
}
