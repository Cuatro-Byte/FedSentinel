"""
Threat/detection endpoints (Contract §24).
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.api.deps import get_threat_service
from backend.database.database import get_db

router = APIRouter()


@router.get("/threats/{run_id}")
def get_threats(run_id: str, db: Session = Depends(get_db)):
    """GET /api/v1/threats/{run_id} — detection results (Contract §24)."""
    service = get_threat_service(db)
    return service.get_threats(run_id)
