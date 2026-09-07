"""
Audit/event endpoints.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.database.repositories import AuditRepository

router = APIRouter()


@router.get("/audit/{run_id}")
def get_audit_events(run_id: str, db: Session = Depends(get_db)):
    """GET /api/v1/audit/{run_id} — audit events."""
    repo = AuditRepository(db)
    events = repo.get_by_run(run_id)
    return [
        {
            "id": e.id,
            "run_id": e.run_id,
            "round_id": e.round_id,
            "client_id": e.client_id,
            "update_id": e.update_id,
            "event_type": e.event_type,
            "severity": e.severity,
            "message": e.message,
            "model_version": e.model_version,
            "detector_version": e.detector_version,
            "timestamp": e.timestamp.isoformat() if e.timestamp else None,
        }
        for e in events
    ]
