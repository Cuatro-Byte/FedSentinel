"""
API response wrappers and error models (Contract §25).

Standard error response format. Do not expose stack traces.
"""

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    """Standard error detail (Contract §25)."""
    code: str
    message: str
    request_id: str | None = None


class ErrorResponse(BaseModel):
    """Standard error response wrapper (Contract §25)."""
    error: ErrorDetail
