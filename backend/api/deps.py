"""
Dependency injection for API routes.

Provides database sessions, services, and adapter instances
via FastAPI's Depends() mechanism.
"""

from sqlalchemy.orm import Session

from backend.config import AppConfig, get_config
from backend.database.database import get_db
from backend.adapters.interfaces import FLCoreInterface, AttackInterface, SentinelInterface
from backend.adapters.registry import configure_adapters, get_adapters
from backend.services.simulation_service import SimulationService
from backend.services.client_service import ClientService
from backend.services.threat_service import ThreatService
from backend.services.impact_service import ImpactService
from backend.services.metrics_service import MetricsService
from backend.services.recovery_service import RecoveryService
_config: AppConfig = get_config()


def get_simulation_service(db: Session) -> SimulationService:
    adapters = get_adapters()
    return SimulationService(
        db,
        adapters.fl_core,
        adapters.attack_engine,
        adapters.sentinel,
        _config,
    )


def get_client_service(db: Session) -> ClientService:
    return ClientService(db)


def get_threat_service(db: Session) -> ThreatService:
    return ThreatService(db)


def get_impact_service(db: Session) -> ImpactService:
    return ImpactService(db)


def get_metrics_service(db: Session) -> MetricsService:
    return MetricsService(db)


def get_recovery_service(db: Session) -> RecoveryService:
    return RecoveryService(db)
