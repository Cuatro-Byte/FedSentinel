"""
core/sentinel/threat_scoring.py

FedSentinel — Threat Scoring Engine
Owner: Person 3 — FedSentinel Detection, Impact & Recovery Intelligence

Converts behavioral signals into a final explainable threat score.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

THREAT_ENGINE_VERSION: str = "threat-v1"
SCHEMA_VERSION: str = "schema-v1"
EPSILON: float = 1e-12

WEIGHT_ANOMALY: float = 0.40
WEIGHT_REPUTATION: float = 0.30
WEIGHT_SIMILARITY: float = 0.20
WEIGHT_HISTORY: float = 0.10


class ThreatScoringError(Exception):
    pass


class ThreatScoringEngine:
    def __init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__module__ + "." + self.__class__.__name__)
        self._logger.debug("ThreatScoringEngine initialised (version=%s)", THREAT_ENGINE_VERSION)

    def compute_threat(self, 
                       feature: dict[str, Any], 
                       statistics: dict[str, Any], 
                       similarity: dict[str, Any], 
                       anomaly: dict[str, Any], 
                       reputation: dict[str, Any]) -> dict[str, Any]:
        """
        Compute the independent composite threat score for a client.
        """
        if not feature or not statistics or not similarity or not anomaly or not reputation:
            raise ThreatScoringError("Missing input signals for threat scoring.")

        try:
            update_id = feature.get("update_id")
            client_id = feature.get("client_id")
            round_id = feature.get("round_id")
            
            if not update_id or not client_id or round_id is None:
                raise ValueError("Missing update_id, client_id, or round_id.")

            anomaly_score = float(anomaly.get("anomaly_score", 0.0))
            rep_score = float(reputation.get("reputation_score", 1.0))
            rep_metrics = reputation.get("reputation_metrics", {})
            rounds_seen = int(rep_metrics.get("rounds_seen", 1))
            consistency = float(rep_metrics.get("consistency_score", 1.0))
            rep_counts = reputation.get("behavior_counts", {})
            suspicious = int(rep_counts.get("suspicious_count", 0))
            malicious = int(rep_counts.get("malicious_count", 0))
            quarantine = int(rep_counts.get("quarantine_count", 0))
            
            sim_anom = float(anomaly.get("signal_breakdown", {}).get("similarity_anomaly", 0.0))
            
            # Threat components
            anomaly_threat = self._clip(anomaly_score)
            reputation_threat = self._clip(1.0 - rep_score)
            similarity_threat = self._clip(sim_anom)
            
            # History threat: penalize counts
            history_penalty = (suspicious + malicious * 2 + quarantine * 3) / 10.0
            history_threat = self._clip(history_penalty)
            
            # Weighted aggregation
            raw_threat = (
                anomaly_threat * WEIGHT_ANOMALY +
                reputation_threat * WEIGHT_REPUTATION +
                similarity_threat * WEIGHT_SIMILARITY +
                history_threat * WEIGHT_HISTORY
            )
            
            threat_score = self._clip(raw_threat)
            
            # Mapping
            threat_level, threat_rank = self._map_threat_level(threat_score)
            
            # Explanations
            total = (anomaly_threat * WEIGHT_ANOMALY +
                     reputation_threat * WEIGHT_REPUTATION +
                     similarity_threat * WEIGHT_SIMILARITY +
                     history_threat * WEIGHT_HISTORY + EPSILON)
                     
            breakdown = {
                "anomaly": float(np.round((anomaly_threat * WEIGHT_ANOMALY) / total * 100.0, 2)),
                "reputation": float(np.round((reputation_threat * WEIGHT_REPUTATION) / total * 100.0, 2)),
                "similarity": float(np.round((similarity_threat * WEIGHT_SIMILARITY) / total * 100.0, 2)),
                "history": float(np.round((history_threat * WEIGHT_HISTORY) / total * 100.0, 2))
            }
            
            # Confidence
            conf = (min(rounds_seen / 10.0, 1.0) * 0.5) + (consistency * 0.5)
            confidence_score = self._clip(conf)
            
            # Summary
            summary = self._generate_summary(anomaly_threat, reputation_threat, similarity_threat, rounds_seen, threat_score, confidence_score)
            
            return {
                "update_id": update_id,
                "client_id": client_id,
                "round_id": round_id,
                "schema_version": SCHEMA_VERSION,
                "threat_version": THREAT_ENGINE_VERSION,
                "threat_score": float(threat_score),
                "threat_level": threat_level,
                "threat_rank": threat_rank,
                "confidence_score": float(confidence_score),
                "threat_breakdown": breakdown,
                "behavior_snapshot": {
                    "reputation_score": rep_score,
                    "anomaly_score": anomaly_score,
                    "consistency_score": consistency,
                    "rounds_seen": rounds_seen
                },
                "threat_summary": summary[:5]
            }
            
        except Exception as e:
            raise ThreatScoringError(f"Failed to compute threat: {e}") from e

    def _clip(self, val: float) -> float:
        if not math.isfinite(val):
            return 1.0
        return float(np.clip(val, 0.0, 1.0))
        
    def _map_threat_level(self, score: float) -> tuple[str, int]:
        if score <= 0.20:
            return "SAFE", 1
        elif score <= 0.40:
            return "LOW", 2
        elif score <= 0.60:
            return "MEDIUM", 3
        elif score <= 0.80:
            return "HIGH", 4
        else:
            return "CRITICAL", 5
            
    def _generate_summary(self, a_t: float, r_t: float, s_t: float, rs: int, ts: float, cs: float) -> list[str]:
        summ = []
        if r_t > 0.6:
            summ.append("Declining reputation history.")
        if a_t > 0.6:
            summ.append("Persistent anomaly history.")
        if s_t > 0.6:
            summ.append("Consensus deviation.")
        if rs == 1 and a_t > 0.5:
            summ.append("Sudden isolated update.")
        if cs > 0.8 and ts <= 0.2:
            summ.append("Stable trustworthy client.")
        return summ
