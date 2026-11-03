"""
Threat service — detection result lookups.

Provides formatted detection data for API consumption.
Does NOT calculate threat scores — those come from P3.
"""

from sqlalchemy.orm import Session

from backend.database.repositories import DetectionRepository


class ThreatService:
    """Detection result lookups for API routes."""

    def __init__(self, db: Session):
        self.detection_repo = DetectionRepository(db)

    def get_threats(self, run_id: str) -> list[dict]:
        """Get all detection results for a simulation run."""
        detections = self.detection_repo.get_by_run(run_id)
        return [
            {
                "update_id": d.update_id,
                "client_id": d.client_id,
                "round_id": d.round_id,
                "threat_score": d.threat_score,
                "threat_level": d.threat_level,
                "action": d.action,
                "anomaly_score": d.anomaly_score,
                "similarity_score": d.similarity_score,
                "reputation_score": d.reputation_score,
                "feature_summary": d.get_feature_summary(),
                "explanation_codes": d.get_explanation_codes(),
                "detector_version": d.detector_version,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            }
            for d in detections
        ]

    def get_threats_by_round(self, run_id: str, round_id: int) -> list[dict]:
        """Get detection results for a specific round."""
        detections = self.detection_repo.get_by_round(run_id, round_id)
        return [
            {
                "update_id": d.update_id,
                "client_id": d.client_id,
                "round_id": d.round_id,
                "threat_score": d.threat_score,
                "threat_level": d.threat_level,
                "action": d.action,
                "anomaly_score": d.anomaly_score,
                "similarity_score": d.similarity_score,
                "reputation_score": d.reputation_score,
                "feature_summary": d.get_feature_summary(),
                "explanation_codes": d.get_explanation_codes(),
                "detector_version": d.detector_version,
            }
            for d in detections
        ]
