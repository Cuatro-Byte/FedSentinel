"""
FedSentinel backend configuration.

Loads simulation, attack, detection, and server settings.
Configuration parameters must be versioned (Contract §33).
"""

from typing import Any
import os
from pydantic_settings import BaseSettings
from pydantic import Field


class SimulationConfig(BaseSettings):
    """Simulation configuration (Contract §32 simulation.yaml)."""
    client_count: int = 20
    rounds: int = 10
    seed: int = 42
    local_epochs: int = 2


class AttackConfig(BaseSettings):
    """Attack configuration (Contract §32 attacks.yaml)."""
    enabled: bool = True
    scenario: str = "backdoor"
    attacker_count: int = 1
    start_round: int = 5
    intensity: float = 0.8
    target_clients: list[str] = Field(default_factory=list)
    targets: Any = None


class DetectionConfig(BaseSettings):
    """Detection configuration (Contract §32 detection.yaml)."""
    safe_threshold: float = 0.50
    malicious_threshold: float = 0.80
    anomaly_weight: float = 0.30
    deviation_weight: float = 0.25
    similarity_weight: float = 0.20
    reputation_weight: float = 0.25


class ImpactConfig(BaseSettings):
    """Impact thresholds (Contract §17.3)."""
    low_threshold: float = 0.0
    medium_threshold: float = 0.25
    high_threshold: float = 0.50
    critical_threshold: float = 0.75


class RecoveryConfig(BaseSettings):
    """Recovery configuration (Contract §20.4)."""
    enabled: bool = True
    accuracy_drop_threshold: float = 0.05
    loss_increase_threshold: float = 0.10


class ServerConfig(BaseSettings):
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    database_url: str = "sqlite:///./fedsentinel.db"
    api_version: str = "v1"
    schema_version: str = "schema-v1"
    frontend_origins: list[str] = Field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.getenv("FRONTEND_ORIGIN", "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000").split(",")
            if origin.strip()
        ]
    )


class AppConfig(BaseSettings):
    """Root application configuration aggregating all sub-configs."""
    server: ServerConfig = Field(default_factory=ServerConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    attack: AttackConfig = Field(default_factory=AttackConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    impact: ImpactConfig = Field(default_factory=ImpactConfig)
    recovery: RecoveryConfig = Field(default_factory=RecoveryConfig)


def get_config() -> AppConfig:
    """Return the application configuration."""
    return AppConfig()
