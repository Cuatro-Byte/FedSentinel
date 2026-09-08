"""
FedSentinel Backend — FastAPI Application Entry Point.

Person 4 owns this module (Contract §5).
Assembles routes, initializes database, configures CORS.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_config
from backend.database.database import init_db

from backend.api.routes.health import router as health_router
from backend.api.routes.simulation import router as simulation_router
from backend.api.routes.clients import router as clients_router
from backend.api.routes.rounds import router as rounds_router
from backend.api.routes.threats import router as threats_router
from backend.api.routes.impact import router as impact_router
from backend.api.routes.metrics import router as metrics_router
from backend.api.routes.recovery import router as recovery_router
from backend.api.routes.audit import router as audit_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("fedsentinel")

config = get_config()


from backend.adapters.registry import configure_adapters
from backend.adapters.p1_fl_adapter import P1FLAdapter
from backend.adapters.p2_attack_adapter import P2AttackAdapter
from backend.adapters.p3_sentinel_adapter import P3SentinelAdapter

from core.federated.client_manager import ClientManager
from core.federated.aggregator import Aggregator
from core.federated.evaluator import Evaluator
from core.federated.checkpoint import CheckpointManager
import torch
import torch.nn as nn

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        
    def forward(self, x):
        return self.fc(x)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — initialize database on startup and configure adapters."""
    logger.info("FedSentinel backend starting...")
    init_db()
    logger.info("Database initialized.")
    
    logger.info("Configuring adapters...")
    
    # Initialize P1 dependencies
    from core.p1_models.model import SimpleCNN
    global_model = SimpleCNN(in_channels=1, num_classes=10)
    client_manager = ClientManager()
    aggregator = Aggregator()
    evaluator = Evaluator()
    checkpoint_manager = CheckpointManager()
    
    p1_adapter = P1FLAdapter(
        global_model=global_model,
        client_manager=client_manager,
        aggregator=aggregator,
        evaluator=evaluator,
        checkpoint_manager=checkpoint_manager,
    )
    p1_adapter.provision_clients(client_count=config.simulation.client_count, seed=config.simulation.seed)
    
    configure_adapters(
        fl_core=p1_adapter,
        attack_engine=P2AttackAdapter(),
        sentinel=P3SentinelAdapter()
    )
    logger.info("Adapters configured.")
    
    yield
    logger.info("FedSentinel backend shutting down.")


app = FastAPI(
    title="FedSentinel",
    description="Federated Learning Defense — Backend API",
    version=config.server.api_version,
    lifespan=lifespan,
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.server.frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all route modules under /api/v1
API_PREFIX = "/api/v1"
app.include_router(health_router, prefix=API_PREFIX, tags=["health"])
app.include_router(simulation_router, prefix=API_PREFIX, tags=["simulation"])
app.include_router(clients_router, prefix=API_PREFIX, tags=["clients"])
app.include_router(rounds_router, prefix=API_PREFIX, tags=["rounds"])
app.include_router(threats_router, prefix=API_PREFIX, tags=["threats"])
app.include_router(impact_router, prefix=API_PREFIX, tags=["impact"])
app.include_router(metrics_router, prefix=API_PREFIX, tags=["metrics"])
app.include_router(recovery_router, prefix=API_PREFIX, tags=["recovery"])
app.include_router(audit_router, prefix=API_PREFIX, tags=["audit"])
