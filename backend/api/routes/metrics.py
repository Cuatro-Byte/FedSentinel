"""
Metrics endpoints (Contract §24).
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.api.deps import get_metrics_service
from backend.database.database import get_db

router = APIRouter()


@router.get("/metrics/{run_id}")
def get_metrics(run_id: str, db: Session = Depends(get_db)):
    """GET /api/v1/metrics/{run_id} — round-by-round metrics (Contract §24)."""
    service = get_metrics_service(db)
    return service.get_metrics(run_id)
