"""
Health check endpoint.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check():
    """GET /api/v1/health — basic health check."""
    return {"status": "healthy", "service": "fedsentinel-backend", "api_version": "v1"}
