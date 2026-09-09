"""
Validation endpoints.

Exposes telemetry produced by ServerValidationGate.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.api.deps import get_validation_service
from backend.api.schemas.validation import ValidationRecordResponse
from backend.database.database import get_db

router = APIRouter()


@router.get("/validation/{run_id}", response_model=list[ValidationRecordResponse])
def get_validation(run_id: str, db: Session = Depends(get_db)):
    """GET /api/v1/validation/{run_id} — server validation gate results."""
    service = get_validation_service(db)
    return service.get_validation_records(run_id)
