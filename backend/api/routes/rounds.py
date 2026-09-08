"""
Round endpoints (Contract §24).
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.database.repositories import RoundRepository

router = APIRouter()


@router.get("/rounds/{run_id}/{round_id}")
def get_round(run_id: str, round_id: int, db: Session = Depends(get_db)):
    """GET /api/v1/rounds/{run_id}/{round_id} — round details (Contract §24)."""
    repo = RoundRepository(db)
    fl_round = repo.get(run_id, round_id)
    if not fl_round:
        raise HTTPException(status_code=404, detail={
            "error": {
                "code": "ROUND_NOT_FOUND",
                "message": f"Round {round_id} not found for run '{run_id}'",
            }
        })
    return {
        "round_id": fl_round.round_id,
        "run_id": fl_round.run_id,
        "model_version": fl_round.model_version,
        "status": fl_round.status,
        "update_count": fl_round.update_count,
        "accepted_count": fl_round.accepted_count,
        "suspicious_count": fl_round.suspicious_count,
        "quarantined_count": fl_round.quarantined_count,
        "downweighted_count": fl_round.downweighted_count,
        "recovery_triggered": fl_round.recovery_triggered,
        "created_at": fl_round.created_at.isoformat() if fl_round.created_at else None,
    }


@router.get("/rounds/{run_id}")
def list_rounds(run_id: str, db: Session = Depends(get_db)):
    """GET /api/v1/rounds/{run_id} — list all rounds for a run."""
    repo = RoundRepository(db)
    rounds = repo.get_by_run(run_id)
    return [
        {
            "round_id": r.round_id,
            "status": r.status,
            "model_version": r.model_version,
            "update_count": r.update_count,
            "accepted_count": r.accepted_count,
            "quarantined_count": r.quarantined_count,
            "recovery_triggered": r.recovery_triggered,
        }
        for r in rounds
    ]
