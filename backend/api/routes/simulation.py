"""
Simulation control endpoints (Contract §24).

Routes are thin — business logic is in SimulationService/Runner.
"""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.api.deps import get_simulation_service
from backend.api.schemas.simulation import SimulationCreateRequest
from backend.database.database import get_db

router = APIRouter()
logger = logging.getLogger("fedsentinel.api.simulation")


@router.post("/simulations")
def create_simulation(request: SimulationCreateRequest,
                      db: Session = Depends(get_db)):
    """POST /api/v1/simulations — start a new simulation (Contract §24)."""
    run_id = f"RUN-{uuid.uuid4().hex[:8].upper()}"

    try:
        service = get_simulation_service(db)
        result = service.start_simulation(
            run_id=run_id,
            scenario=request.scenario,
            client_count=request.client_count,
            rounds=request.rounds,
            attack_enabled=request.attack_enabled,
        )
        return result
    except RuntimeError as e:
        logger.error("Simulation adapters are not configured: %s", e)
        raise HTTPException(status_code=503, detail={
            "error": {
                "code": "ADAPTERS_NOT_CONFIGURED",
                "message": "Simulation adapters are not configured",
            }
        })
    except Exception:
        logger.exception("Simulation %s failed", run_id)
        raise HTTPException(status_code=500, detail={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "The simulation could not be completed",
            }
        })


@router.get("/simulations/{run_id}")
def get_simulation(run_id: str, db: Session = Depends(get_db)):
    """GET /api/v1/simulations/{run_id} — simulation status (Contract §24)."""
    service = get_simulation_service(db)
    result = service.get_simulation(run_id)
    if not result:
        raise HTTPException(status_code=404, detail={
            "error": {
                "code": "RUN_NOT_FOUND",
                "message": f"Simulation run '{run_id}' not found",
            }
        })
    return result


@router.get("/simulations")
def list_simulations(db: Session = Depends(get_db)):
    """GET /api/v1/simulations — list all simulation runs."""
    service = get_simulation_service(db)
    return service.list_simulations()
