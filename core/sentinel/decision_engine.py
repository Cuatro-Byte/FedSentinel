"""
core/sentinel/decision_engine.py

FedSentinel — Decision Engine
Owner: Person 3 — FedSentinel Detection, Impact & Recovery Intelligence

Converts intelligence into actionable policy recommendations.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

DECISION_ENGINE_VERSION: str = "decision-v1"
SCHEMA_VERSION: str = "schema-v1"
EPSILON: float = 1e-12

# Policy Rule Thresholds
ROLLBACK_THREAT_THRESHOLD = 0.80
ROLLBACK_IMPACT_THRESHOLD = 0.80

QUARANTINE_THREAT_THRESHOLD = 0.70
QUARANTINE_IMPACT_THRESHOLD = 0.70
QUARANTINE_HISTORY_THRESHOLD = 0.50

FLAG_THREAT_THRESHOLD = 0.50
FLAG_IMPACT_THRESHOLD = 0.50

MONITOR_THREAT_THRESHOLD = 0.30
MONITOR_IMPACT_THRESHOLD = 0.30


class DecisionEngineError(Exception):
    pass


class DecisionEngine:
    def __init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__module__ + "." + self.__class__.__name__)
        self._logger.debug("DecisionEngine initialised (version=%s)", DECISION_ENGINE_VERSION)

    def compute_decision(self,
                         feature: dict[str, Any],
                         threat: dict[str, Any],
                         impact: dict[str, Any],
                         reputation: dict[str, Any],
                         anomaly: dict[str, Any]) -> dict[str, Any]:
        """
        Compute the recommended security action and explainability.
        """
        return self._evaluate_policy(feature, threat, impact, reputation, anomaly)

    def simulate_decision(self,
                          feature: dict[str, Any],
                          threat: dict[str, Any],
                          impact: dict[str, Any],
                          reputation: dict[str, Any],
                          anomaly: dict[str, Any]) -> dict[str, Any]:
        """
        Simulation mode: compute decision without mutating any history or state.
        (Currently purely functional as mutation belongs to Phase 9).
        """
        return self._evaluate_policy(feature, threat, impact, reputation, anomaly)

    def _evaluate_policy(self,
                         feature: dict[str, Any],
                         threat: dict[str, Any],
                         impact: dict[str, Any],
                         reputation: dict[str, Any],
                         anomaly: dict[str, Any]) -> dict[str, Any]:

        if feature is None or threat is None or impact is None or reputation is None or anomaly is None:
            raise DecisionEngineError("Missing inputs for decision evaluation.")

        try:
            update_id = feature.get("update_id")
            client_id = feature.get("client_id")
            round_id = feature.get("round_id")

            if not update_id or not client_id or round_id is None:
                raise ValueError("Missing update_id, client_id, or round_id.")

            threat_score = float(threat.get("threat_score", 0.0))
            confidence_score = float(threat.get("confidence_score", 1.0))
            
            impact_score = float(impact.get("impact_score", 0.0))
            historical_damage_risk = float(impact.get("historical_damage_risk", 0.0))
            
            reputation_score = float(reputation.get("reputation_score", 1.0))
            rep_metrics = reputation.get("reputation_metrics", {})
            consistency_score = float(rep_metrics.get("consistency_score", 1.0))

            # NaN/Inf guards
            if not math.isfinite(threat_score) or not math.isfinite(impact_score):
                threat_score = 1.0
                impact_score = 1.0

            # Confidence Gate Calculation
            threat_stability = confidence_score
            decision_confidence = self._clip(
                (threat_score * 0.2) +
                (confidence_score * 0.3) +
                (consistency_score * 0.3) +
                ((1.0 - historical_damage_risk) * 0.2)
            )

            # Policy Evaluation (Priority Order)
            recommended_action = "ACCEPT"
            matched_policy = "ACCEPT_RULES"
            matched_threshold = 0.0
            escalation_level = "NONE"

            reason_codes = []
            triggered_rules = []

            # 5. ROLLBACK_RECOMMENDED
            if threat_score >= ROLLBACK_THREAT_THRESHOLD and impact_score >= ROLLBACK_IMPACT_THRESHOLD:
                recommended_action = "ROLLBACK_RECOMMENDED"
                matched_policy = "ROLLBACK_RULES"
                matched_threshold = ROLLBACK_THREAT_THRESHOLD
                escalation_level = "CRITICAL"
                reason_codes.append("CRITICAL_THREAT_IMPACT")
                triggered_rules.append("ROLLBACK_THREAT_AND_IMPACT")

            # 4. QUARANTINE
            elif threat_score >= QUARANTINE_THREAT_THRESHOLD or (impact_score >= QUARANTINE_IMPACT_THRESHOLD and historical_damage_risk >= QUARANTINE_HISTORY_THRESHOLD):
                recommended_action = "QUARANTINE"
                matched_policy = "QUARANTINE_RULES"
                matched_threshold = QUARANTINE_THREAT_THRESHOLD
                escalation_level = "HIGH"
                if threat_score >= QUARANTINE_THREAT_THRESHOLD:
                    reason_codes.append("HIGH_THREAT")
                    triggered_rules.append("QUARANTINE_THREAT")
                else:
                    reason_codes.append("HIGH_IMPACT_HISTORY")
                    triggered_rules.append("QUARANTINE_IMPACT_AND_HISTORY")

            # 3. FLAG
            elif threat_score >= FLAG_THREAT_THRESHOLD or impact_score >= FLAG_IMPACT_THRESHOLD:
                recommended_action = "FLAG"
                matched_policy = "FLAG_RULES"
                matched_threshold = FLAG_THREAT_THRESHOLD
                escalation_level = "MEDIUM"
                reason_codes.append("ELEVATED_RISK")
                triggered_rules.append("FLAG_THREAT_OR_IMPACT")

            # 2. MONITOR
            elif threat_score >= MONITOR_THREAT_THRESHOLD or impact_score >= MONITOR_IMPACT_THRESHOLD:
                recommended_action = "MONITOR"
                matched_policy = "MONITOR_RULES"
                matched_threshold = MONITOR_THREAT_THRESHOLD
                escalation_level = "LOW"
                reason_codes.append("WATCHLIST_ACTIVITY")
                triggered_rules.append("MONITOR_THREAT_OR_IMPACT")

            else:
                reason_codes.append("NOMINAL_BEHAVIOR")
                triggered_rules.append("DEFAULT_ACCEPT")

            # Explanations sorting and limit
            reason_codes = sorted(list(set(reason_codes)))[:5]
            triggered_rules = sorted(list(set(triggered_rules)))[:5]

            # Risk Matrix
            risk_matrix = self._map_risk_matrix(threat_score, impact_score)

            return {
                "update_id": update_id,
                "client_id": client_id,
                "round_id": round_id,
                "schema_version": SCHEMA_VERSION,
                "decision_version": DECISION_ENGINE_VERSION,
                "recommended_action": recommended_action,
                "decision_confidence": float(decision_confidence),
                "escalation_level": escalation_level,
                "risk_matrix": risk_matrix,
                "decision_explanation": {
                    "reason_codes": reason_codes,
                    "triggered_rules": triggered_rules,
                    "supporting_metrics": {
                        "threat_score": float(threat_score),
                        "impact_score": float(impact_score),
                        "reputation_score": float(reputation_score),
                        "confidence_score": float(confidence_score)
                    }
                },
                "policy_evaluation": {
                    "matched_policy": matched_policy,
                    "matched_threshold": float(matched_threshold)
                }
            }

        except Exception as e:
            raise DecisionEngineError(f"Failed to compute decision: {e}") from e

    def _clip(self, val: float) -> float:
        if not math.isfinite(val):
            return 1.0
        return float(np.clip(val, 0.0, 1.0))

    def _map_risk_matrix(self, threat: float, impact: float) -> str:
        if threat >= 0.5 and impact >= 0.5:
            return "CRITICAL"
        elif threat >= 0.5 and impact < 0.5:
            return "ELEVATED"
        elif threat < 0.5 and impact >= 0.5:
            return "WATCHLIST"
        else:
            return "LOW_RISK"
