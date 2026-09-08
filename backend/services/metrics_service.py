"""
Metrics service — aggregated metric queries.

Provides round-by-round metrics for API consumption.
Metrics are assembled by P4 from data produced by P1, P2, P3.
"""

from sqlalchemy.orm import Session

from backend.database.repositories import MetricRepository


class MetricsService:
    """Metrics lookups for API routes."""

    def __init__(self, db: Session):
        self.metric_repo = MetricRepository(db)

    def get_metrics(self, run_id: str) -> list[dict]:
        """Get round-by-round metrics for a simulation run."""
        records = self.metric_repo.get_by_run(run_id)
        return [
            {
                "run_id": m.run_id,
                "round_id": m.round_id,
                "model_version": m.model_version,
                "accuracy": m.accuracy,
                "loss": m.loss,
                "attack_success_rate": m.attack_success_rate,
                "malicious_updates": m.malicious_updates,
                "suspicious_updates": m.suspicious_updates,
                "quarantined_updates": m.quarantined_updates,
                "accepted_updates": m.accepted_updates,
                "downweighted_updates": m.downweighted_updates,
                "recovery_triggered": m.recovery_triggered,
                "recovery_count": m.recovery_count,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in records
        ]
