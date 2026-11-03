"""
Impact endpoints (Contract §24).
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.api.deps import get_impact_service
from backend.database.database import get_db

router = APIRouter()


@router.get("/impact/{run_id}")
def get_impacts(run_id: str, db: Session = Depends(get_db)):
    """GET /api/v1/impact/{run_id} — impact results."""
    service = get_impact_service(db)
    return service.get_impacts(run_id)
