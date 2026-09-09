"""
Validation service — server validation telemetry lookups.

Provides formatted validation gate telemetry for API consumption.
Does NOT compute or manipulate validation metrics.
"""

from sqlalchemy.orm import Session

from backend.database.repositories import ValidationRepository


class ValidationService:
    """Validation telemetry lookups for API routes."""

    def __init__(self, db: Session):
        self.val_repo = ValidationRepository(db)

    def get_validation_records(self, run_id: str) -> list[dict]:
        """Get all validation records for a simulation run."""
        records = self.val_repo.get_by_run(run_id)
        return [
            {
                "run_id": r.run_id,
                "round_id": r.round_id,
                "model_version": r.model_version,
                "validation_loss": r.validation_loss,
                "validation_accuracy": r.validation_accuracy,
                "loss_spiked": r.loss_spiked,
                "baseline_loss": r.baseline_loss,
                "baseline_accuracy": r.baseline_accuracy,
                "loss_delta": r.loss_delta,
                "accuracy_delta": r.accuracy_delta,
                "validation_status": r.validation_status,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ]
