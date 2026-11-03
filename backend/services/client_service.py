"""
Client service — client data queries.

Provides formatted client data for API consumption.
Does NOT contain detection or reputation algorithms.
"""

from sqlalchemy.orm import Session

from backend.database.repositories import ClientRepository


class ClientService:
    """Client data lookups for API routes."""

    def __init__(self, db: Session):
        self.client_repo = ClientRepository(db)

    def get_clients(self, run_id: str) -> list[dict]:
        """Get all clients for a simulation run."""
        clients = self.client_repo.get_by_run(run_id)
        return [
            {
                "client_id": c.client_id,
                "reputation_score": c.reputation_score,
                "latest_threat_score": c.last_threat_score,
                "latest_impact_score": c.last_impact_score,
                "action": c.last_action,
                "rounds_participated": c.rounds_participated,
                "suspicious_count": c.suspicious_count,
                "malicious_count": c.malicious_count,
                "quarantine_count": c.quarantine_count,
            }
            for c in clients
        ]

    def get_client(self, run_id: str, client_id: str) -> dict | None:
        """Get a single client's state."""
        c = self.client_repo.get(run_id, client_id)
        if not c:
            return None
        return {
            "client_id": c.client_id,
            "reputation_score": c.reputation_score,
            "latest_threat_score": c.last_threat_score,
            "latest_impact_score": c.last_impact_score,
            "action": c.last_action,
            "rounds_participated": c.rounds_participated,
            "suspicious_count": c.suspicious_count,
            "malicious_count": c.malicious_count,
            "quarantine_count": c.quarantine_count,
        }
