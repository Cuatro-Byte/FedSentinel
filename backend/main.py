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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — initialize database on startup."""
    logger.info("FedSentinel backend starting...")
    init_db()
    logger.info("Database initialized.")
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
    allow_origins=["*"],
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
