"""
Recovery service — recovery event queries.

Provides formatted recovery data for API consumption.
Does NOT implement recovery intelligence — that is P3's responsibility.
"""

from sqlalchemy.orm import Session

from backend.database.repositories import RecoveryRepository


class RecoveryService:
    """Recovery event lookups for API routes."""

    def __init__(self, db: Session):
        self.recovery_repo = RecoveryRepository(db)

    def get_recoveries(self, run_id: str) -> list[dict]:
        """Get all recovery events for a simulation run."""
        records = self.recovery_repo.get_by_run(run_id)
        return [
            {
                "recovery_id": r.recovery_id,
                "run_id": r.run_id,
                "round_id": r.round_id,
                "trigger": r.trigger,
                "affected_update_ids": r.get_affected_update_ids(),
                "excluded_client_ids": r.get_excluded_client_ids(),
                "previous_model_version": r.previous_model_version,
                "recovered_model_version": r.recovered_model_version,
                "before_accuracy": r.before_accuracy,
                "after_accuracy": r.after_accuracy,
                "before_loss": r.before_loss,
                "after_loss": r.after_loss,
                "recovery_status": r.recovery_status,
                "recovery_version": r.recovery_version,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ]
