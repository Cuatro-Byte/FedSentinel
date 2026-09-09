"""
API-specific request/response schemas.

These are API transport schemas ONLY. They do NOT duplicate
the canonical models in core/models/.
"""

from typing import Any
from pydantic import BaseModel, Field, model_validator


VALID_SCENARIOS = {
    "normal",
    "model_poisoning",
    "label_poisoning",
    "label_flipping",
    "backdoor",
    "mixed_attack",
    "sleeper",
    "sybil",
}


class SimulationCreateRequest(BaseModel):
    """Request body for POST /api/v1/simulations (Contract §24)."""
    client_count: int = Field(default=20, gt=0, le=100)
    rounds: int = Field(default=10, gt=0, le=100)
    scenario: str = Field(default="backdoor")
    attack_enabled: bool = True
    attacker_count: int = Field(default=1, ge=0, le=100)
    intensity: float | None = Field(default=None, ge=0.0, le=100.0)
    start_round: int = Field(default=1, ge=1, le=100)
    targets: Any = None
    seed: int | None = Field(default=42, ge=0, le=2147483647)
    background: bool = Field(default=False)

    @model_validator(mode="after")
    def validate_simulation_parameters(self) -> "SimulationCreateRequest":
        if self.scenario not in VALID_SCENARIOS:
            raise ValueError(
                f"Invalid scenario '{self.scenario}'. Valid scenarios are: {sorted(VALID_SCENARIOS)}"
            )
        if self.start_round > self.rounds:
            raise ValueError(
                f"start_round ({self.start_round}) cannot be greater than rounds ({self.rounds})"
            )
        if self.attacker_count > self.client_count:
            raise ValueError(
                f"attacker_count ({self.attacker_count}) cannot exceed client_count ({self.client_count})"
            )
        return self


class SimulationStatusResponse(BaseModel):
    """Response for simulation status."""
    run_id: str
    simulation_id: str | None = None
    status: str

