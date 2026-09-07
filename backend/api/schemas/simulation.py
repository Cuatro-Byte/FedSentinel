"""
API-specific request/response schemas.

These are API transport schemas ONLY. They do NOT duplicate
the canonical models in core/models/.
"""

from pydantic import BaseModel, Field


class SimulationCreateRequest(BaseModel):
    """Request body for POST /api/v1/simulations (Contract §24)."""
    client_count: int = Field(default=20, gt=0)
    rounds: int = Field(default=10, gt=0)
    scenario: str = "backdoor"
    attack_enabled: bool = True


class SimulationStatusResponse(BaseModel):
    """Response for simulation status."""
    run_id: str
    status: str
