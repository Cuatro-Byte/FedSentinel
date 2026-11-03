"""
core/sentinel/recovery_engine.py

FedSentinel — Recovery Intelligence Engine
Owner: Person 3 — FedSentinel Detection, Impact & Recovery Intelligence

Recommends how Sentinel should recover from malicious/suspicious updates.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

RECOVERY_ENGINE_VERSION: str = "recovery-v1"
SCHEMA_VERSION: str = "schema-v1"
EPSILON: float = 1e-12

# Policy Rules (Constants)
ROLLBACK_POLICY = "ROLLBACK_POLICY"
QUARANTINE_POLICY = "QUARANTINE_POLICY"
ISOLATE_POLICY = "ISOLATE_POLICY"
MONITOR_POLICY = "MONITOR_POLICY"
ACCEPT_POLICY = "ACCEPT_POLICY"


class RecoveryEngineError(Exception):
    pass


class RecoveryEngine:
    def __init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__module__ + "." + self.__class__.__name__)
        self._logger.debug("RecoveryEngine initialised (version=%s)", RECOVERY_ENGINE_VERSION)

    def compute_recovery(self,
                         decision: dict[str, Any],
                         threat: dict[str, Any],
                         impact: dict[str, Any],
                         reputation: dict[str, Any]) -> dict[str, Any]:
        """
        Compute the recommended recovery plan.
        """
        return self._generate_recovery_plan(decision, threat, impact, reputation)

    def simulate_recovery(self,
                          decision: dict[str, Any],
                          threat: dict[str, Any],
                          impact: dict[str, Any],
                          reputation: dict[str, Any]) -> dict[str, Any]:
        """
        Simulate recovery generation without side-effects.
        """
        return self._generate_recovery_plan(decision, threat, impact, reputation)

    def _generate_recovery_plan(self,
                                decision: dict[str, Any],
                                threat: dict[str, Any],
                                impact: dict[str, Any],
                                reputation: dict[str, Any]) -> dict[str, Any]:
        if decision is None or threat is None or impact is None or reputation is None:
            raise RecoveryEngineError("Missing inputs for recovery evaluation.")

        try:
            update_id = decision.get("update_id")
            client_id = decision.get("client_id")
            round_id = decision.get("round_id")

            if not update_id or not client_id or round_id is None:
                raise ValueError("Missing update_id, client_id, or round_id.")

            decision_action = decision.get("recommended_action", "ACCEPT")
            decision_confidence = float(decision.get("decision_confidence", 1.0))
            
            threat_score = float(threat.get("threat_score", 0.0))
            confidence_score = float(threat.get("confidence_score", 1.0))
            
            impact_score = float(impact.get("impact_score", 0.0))
            historical_damage_risk = float(impact.get("historical_damage_risk", 0.0))
            
            rep_metrics = reputation.get("reputation_metrics", {})
            consistency_score = float(rep_metrics.get("consistency_score", 1.0))

            # NaN/Inf guards
            if not math.isfinite(threat_score) or not math.isfinite(impact_score):
                threat_score = 1.0
                impact_score = 1.0

            threat_stability = confidence_score

            # Recovery Confidence
            raw_confidence = (
                decision_confidence +
                threat_stability +
                consistency_score +
                (1.0 - historical_damage_risk)
            ) / 4.0
            recovery_confidence = self._clip(raw_confidence)

            # Determine Recommendations
            rollback_recommended = False
            quarantine_recommended = False
            isolation_recommended = False

            # ROLLBACK Logic
            # CRITICAL threat, HIGH/SEVERE impact, High confidence, persistent threat (history > 0.5 or consistency < 0.5)
            if threat_score >= 0.8 and impact_score >= 0.6 and decision_confidence >= 0.7 and historical_damage_risk >= 0.5:
                rollback_recommended = True
            elif decision_action == "ROLLBACK_RECOMMENDED":
                rollback_recommended = True

            # QUARANTINE Logic
            # HIGH threat, HIGH impact, Confidence sufficient
            if threat_score >= 0.6 and impact_score >= 0.6 and decision_confidence >= 0.5:
                quarantine_recommended = True
            elif decision_action == "QUARANTINE":
                quarantine_recommended = True

            # ISOLATION Logic
            # Temporary isolation for medium-risk
            if threat_score >= 0.4 and impact_score >= 0.4 and not quarantine_recommended and not rollback_recommended:
                isolation_recommended = True
            elif decision_action == "FLAG":
                isolation_recommended = True

            # Priority Resolution
            primary_action = "ACCEPT"
            triggered_policy = ACCEPT_POLICY
            expected_effect = "Client update processed normally."
            timeline = "Immediate"
            
            # Monitoring Plan Init
            monitoring_frequency = "standard"
            monitoring_duration = "0"
            signals_to_monitor = []
            re_evaluation_round = round_id + 1

            if rollback_recommended:
                primary_action = "ROLLBACK"
                triggered_policy = ROLLBACK_POLICY
                expected_effect = "Revert global model state to prevent critical damage."
                timeline = "Immediate"
                monitoring_frequency = "continuous"
                monitoring_duration = "indefinite"
                signals_to_monitor = ["model_drift", "client_history"]
                re_evaluation_round = round_id + 10
            elif quarantine_recommended:
                primary_action = "QUARANTINE"
                triggered_policy = QUARANTINE_POLICY
                expected_effect = "Block client updates while tracking historical behavior."
                timeline = "Immediate"
                monitoring_frequency = "high"
                monitoring_duration = "10_rounds"
                signals_to_monitor = ["reputation_decay", "anomaly_magnitude"]
                re_evaluation_round = round_id + 5
            elif isolation_recommended:
                primary_action = "ISOLATE_CLIENT"
                triggered_policy = ISOLATE_POLICY
                expected_effect = "Temporarily segregate client updates for review."
                timeline = "Next Round"
                monitoring_frequency = "elevated"
                monitoring_duration = "5_rounds"
                signals_to_monitor = ["consistency", "distribution_drift"]
                re_evaluation_round = round_id + 3
            elif decision_action == "MONITOR":
                primary_action = "MONITOR"
                triggered_policy = MONITOR_POLICY
                expected_effect = "Increase tracking sensitivity without blocking."
                timeline = "Next 3 Rounds"
                monitoring_frequency = "elevated"
                monitoring_duration = "3_rounds"
                signals_to_monitor = ["anomaly_trend"]
                re_evaluation_round = round_id + 1

            secondary_actions = []
            if rollback_recommended and not primary_action == "QUARANTINE":
                secondary_actions.append("QUARANTINE_CLIENT")
            if quarantine_recommended and not primary_action == "ISOLATE_CLIENT":
                secondary_actions.append("ISOLATE_CLIENT")

            # Explainability
            why_this_action = f"Primary decision logic dictated {decision_action} resulting in {primary_action}."
            explanation = {
                "why_this_action": why_this_action,
                "triggered_policy": triggered_policy,
                "expected_effect": expected_effect,
                "supporting_metrics": {
                    "threat_score": float(threat_score),
                    "impact_score": float(impact_score),
                    "decision_confidence": float(decision_confidence),
                    "recovery_confidence": float(recovery_confidence)
                }
            }

            monitoring_plan = {
                "monitoring_frequency": monitoring_frequency,
                "monitoring_duration": monitoring_duration,
                "signals_to_monitor": signals_to_monitor,
                "re_evaluation_round": int(re_evaluation_round)
            }

            return {
                "update_id": update_id,
                "client_id": client_id,
                "round_id": round_id,
                "schema_version": SCHEMA_VERSION,
                "recovery_version": RECOVERY_ENGINE_VERSION,
                "primary_action": primary_action,
                "secondary_actions": secondary_actions,
                "rollback_recommended": rollback_recommended,
                "quarantine_recommended": quarantine_recommended,
                "isolation_recommended": isolation_recommended,
                "monitoring_plan": monitoring_plan,
                "recovery_confidence": float(recovery_confidence),
                "recovery_timeline": timeline,
                "recovery_explanation": explanation
            }

        except Exception as e:
            raise RecoveryEngineError(f"Failed to compute recovery: {e}") from e

    def _clip(self, val: float) -> float:
        if not math.isfinite(val):
            return 1.0
        return float(np.clip(val, 0.0, 1.0))
