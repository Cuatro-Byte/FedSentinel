"""
core/sentinel/reputation.py

FedSentinel — Client Reputation Engine
Owner: Person 3 — FedSentinel Detection, Impact & Recovery Intelligence

This module maintains the persistent historical trust profile of clients
based on anomaly signals across federated learning rounds.

Contract reference: FEDSENTINEL_NEW_TEAM_ENGINEERING_CONTRACT_v2.md
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

REPUTATION_ENGINE_VERSION: str = "reputation-v1"
SCHEMA_VERSION: str = "schema-v1"
EPSILON: float = 1e-12

HISTORY_WINDOW: int = 5
DECAY_COEF: float = 0.2
RECOVERY_COEF: float = 0.05
ANOMALY_THRESHOLD: float = 0.5


class ReputationEngineError(Exception):
    """
    Raised when the Reputation Engine cannot safely apply updates
    (e.g., out-of-order rounds, duplicate updates, missing keys).
    """


@dataclass
class ClientReputationState:
    """
    Mandatory dataclass tracking persistent client state.
    """
    client_id: str
    reputation_score: float = 1.0
    rounds_seen: int = 0
    suspicious_count: int = 0
    malicious_count: int = 0
    quarantine_count: int = 0
    last_anomaly_score: float = 0.0
    reputation_history: list[dict[str, Any]] = field(default_factory=list)
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ReputationEngine:
    """
    Stateful engine mapping anomaly outputs to long-term client reputation.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(
            self.__class__.__module__ + "." + self.__class__.__name__
        )
        self._logger.debug(
            "ReputationEngine initialised (version=%s schema=%s)",
            REPUTATION_ENGINE_VERSION,
            SCHEMA_VERSION,
        )
        # Store persistent states in memory for the duration of the Sentinel process.
        self._client_states: dict[str, ClientReputationState] = {}

    def update_reputation(self, client_id: str, anomaly_output: dict[str, Any]) -> dict[str, Any]:
        """
        Update the reputation profile for a client given their latest anomaly output.

        Parameters
        ----------
        client_id:
            The unique identifier for the client.
        anomaly_output:
            The full dictionary output from the AnomalyDetector.

        Returns
        -------
        dict[str, Any]
            The client's reputation state dictionary matching Phase 5 output contract.

        Raises
        ------
        ReputationEngineError
            If inputs are missing, NaN/Inf, or updates are out-of-order.
        """
        if not client_id:
            raise ReputationEngineError("Missing client_id.")
        if not anomaly_output:
            raise ReputationEngineError("Missing anomaly_output.")

        round_id = anomaly_output.get("round_id")
        if round_id is None:
            raise ReputationEngineError(f"Missing round_id for client {client_id}")

        anomaly_score_raw = anomaly_output.get("anomaly_score")
        if anomaly_score_raw is None or not math.isfinite(anomaly_score_raw):
            raise ReputationEngineError(f"Invalid anomaly score for client {client_id}")
        
        anomaly_score = float(anomaly_score_raw)

        # 1. Fetch or Initialize State
        state = self._client_states.get(client_id)
        if not state:
            state = ClientReputationState(client_id=client_id)
            self._client_states[client_id] = state

        # 2. Reject duplicate or out-of-order rounds
        if state.reputation_history:
            last_round = state.reputation_history[-1].get("round_id", -1)
            if round_id <= last_round:
                raise ReputationEngineError(
                    f"Out of order or duplicate round update for client {client_id}: "
                    f"attempted {round_id}, last was {last_round}"
                )

        # 3. Apply Decay or Recovery
        if anomaly_score >= ANOMALY_THRESHOLD:
            # Decay: drops based on how anomalous it is
            delta = - (anomaly_score * DECAY_COEF)
        else:
            # Recovery: recovers slowly if anomaly is low
            delta = (1.0 - anomaly_score) * RECOVERY_COEF

        new_rep = float(np.clip(state.reputation_score + delta, 0.0, 1.0))
        
        # 4. Update State
        state.rounds_seen += 1
        state.reputation_score = new_rep
        state.last_anomaly_score = anomaly_score
        state.updated_at = datetime.now(timezone.utc)

        # We temporarily create the new snapshot minus metrics for calculation
        # To compute rolling metrics, we need the history including this step.
        recent_anomalies = [snap["anomaly_score"] for snap in state.reputation_history[-HISTORY_WINDOW + 1:]] + [anomaly_score]
        recent_reputations = [snap["reputation_score"] for snap in state.reputation_history[-HISTORY_WINDOW + 1:]] + [new_rep]

        # 5. Compute Historical Metrics
        rolling_avg_anom = float(np.mean(recent_anomalies))
        rolling_avg_rep = float(np.mean(recent_reputations))
        
        anom_trend = self._compute_trend(recent_anomalies)
        rep_trend = self._compute_trend(recent_reputations)
        
        # Consistency is inverse of the standard deviation of recent anomaly scores
        std_anom = float(np.std(recent_anomalies))
        consistency_score = float(np.clip(1.0 - std_anom, 0.0, 1.0))

        # 6. Determine Flags
        flags = []
        if state.rounds_seen == 1:
            flags.append("NEW_CLIENT")
        else:
            if consistency_score > 0.8 and new_rep > 0.8:
                flags.append("STABLE_CLIENT")
            if rep_trend < -0.01:
                flags.append("DECLINING_REPUTATION")
            if rep_trend > 0.01:
                flags.append("RECOVERING_REPUTATION")
            if rolling_avg_anom > 0.7:
                flags.append("HIGH_ANOMALY_HISTORY")

        # 7. Append Snapshot
        timestamp_str = state.updated_at.isoformat()
        snapshot = {
            "round_id": round_id,
            "anomaly_score": anomaly_score,
            "reputation_score": new_rep,
            "consistency_score": consistency_score,
            "flags": flags,
            "timestamp": timestamp_str
        }
        # Immutable append
        state.reputation_history.append(snapshot)

        self._logger.info(
            "Updated reputation for client %s to %f (round %s)",
            client_id, new_rep, round_id
        )

        # 8. Return strictly formatted dictionary
        return {
            "client_id": client_id,
            "schema_version": SCHEMA_VERSION,
            "reputation_version": REPUTATION_ENGINE_VERSION,
            "reputation_score": new_rep,
            "reputation_metrics": {
                "rounds_seen": state.rounds_seen,
                "rolling_average_anomaly": rolling_avg_anom,
                "rolling_average_reputation": rolling_avg_rep,
                "consistency_score": consistency_score,
                "anomaly_trend": anom_trend,
                "reputation_trend": rep_trend
            },
            "behavior_counts": {
                "suspicious_count": state.suspicious_count,
                "malicious_count": state.malicious_count,
                "quarantine_count": state.quarantine_count
            },
            "reputation_flags": flags,
            "last_anomaly_score": anomaly_score,
            "latest_snapshot": snapshot
        }

    def _compute_trend(self, values: list[float]) -> float:
        """
        Compute the linear trend (slope) over a list of values.
        """
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        y = np.array(values)
        # Using polyfit to find slope of line of best fit
        slope = float(np.polyfit(x, y, 1)[0])
        
        # Guard against minor floating point drift near 0
        if abs(slope) < EPSILON:
            return 0.0
        return slope
