"""
Recovery endpoints (Contract §24).
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.api.deps import get_recovery_service
from backend.database.database import get_db

router = APIRouter()


@router.get("/recovery/{run_id}")
def get_recoveries(run_id: str, db: Session = Depends(get_db)):
    """GET /api/v1/recovery/{run_id} — recovery events (Contract §24)."""
    service = get_recovery_service(db)
    return service.get_recoveries(run_id)
