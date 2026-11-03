"""
Client endpoints (Contract §24).
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.api.deps import get_client_service
from backend.database.database import get_db

router = APIRouter()


@router.get("/clients/{run_id}")
def get_clients(run_id: str, db: Session = Depends(get_db)):
    """GET /api/v1/clients/{run_id} — client-level state (Contract §24)."""
    service = get_client_service(db)
    clients = service.get_clients(run_id)
    if not clients:
        raise HTTPException(status_code=404, detail={
            "error": {
                "code": "RUN_NOT_FOUND",
                "message": f"No clients found for run '{run_id}'",
            }
        })
    return clients
