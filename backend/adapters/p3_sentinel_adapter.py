import logging
from typing import Dict, List
from datetime import datetime

from backend.adapters.interfaces import SentinelInterface
from core.models.model_update import ModelUpdate
from core.models.detection_result import DetectionResult
from core.models.impact_result import ImpactResult
from core.models.recovery_result import RecoveryResult
from core.models.enums import ResponseAction, ThreatLevel, ImpactLevel, RecoveryStatus
from core.sentinel.sentinel import Sentinel

logger = logging.getLogger(__name__)

class P3SentinelAdapter(SentinelInterface):
    """
    Adapter integrating Person 3's Sentinel intelligence engine into the P4 orchestrator.
    Translates P3's internal dictionaries into canonical P4 application models.
    """

    def __init__(self):
        self.sentinel = Sentinel()
        self._round_cache: dict[int, list[dict]] = {}

    def reset(self) -> None:
        """Reset internal Sentinel instance and caches for a new simulation run."""
        self.sentinel = Sentinel()
        self._round_cache = {}

    def detect(self, updates: list[ModelUpdate], round_id: int) -> list[DetectionResult]:
        """
        Run the complete P3 Sentinel pipeline (Phases 1-9) and cache the results.
        Returns the parsed DetectionResult list.
        """
        # Execute the full pipeline via Sentinel
        results = self.sentinel.run_round(updates, round_id)
        self._round_cache[round_id] = results
        
        detections = []
        for res in results:
            dr_dict = res.get("client_result", {}).get("detection_result", {})
            pipeline_status = res.get("pipeline_metadata", {}).get("pipeline_status", "SUCCESS")
            action_str = dr_dict.get("action")

            if pipeline_status == "FAILED" or not action_str:
                action = ResponseAction.QUARANTINE
                threat_level = ThreatLevel.MALICIOUS
                threat_score = 1.0
            elif action_str == "ACCEPT":
                action = ResponseAction.ACCEPT
                threat_level_str = dr_dict.get("threat_level", "SAFE")
                threat_level = ThreatLevel.SAFE if threat_level_str == "SAFE" else (
                    ThreatLevel.SUSPICIOUS if threat_level_str == "SUSPICIOUS" else ThreatLevel.MALICIOUS
                )
                threat_score = dr_dict.get("threat_score", 0.0)
            elif action_str == "DOWN_WEIGHT":
                action = ResponseAction.DOWN_WEIGHT
                threat_level_str = dr_dict.get("threat_level", "SUSPICIOUS")
                threat_level = ThreatLevel.SUSPICIOUS if threat_level_str == "SUSPICIOUS" else ThreatLevel.MALICIOUS
                threat_score = dr_dict.get("threat_score", 0.5)
            else:
                action = ResponseAction.QUARANTINE
                threat_level = ThreatLevel.MALICIOUS
                threat_score = dr_dict.get("threat_score", 1.0)
            
            detection = DetectionResult(
                update_id=dr_dict.get("update_id", "unknown"),
                client_id=dr_dict.get("client_id", "unknown"),
                round_id=dr_dict.get("round_id", round_id),
                threat_score=dr_dict.get("threat_score", 0.0),
                threat_level=threat_level,
                action=action,
                feature_summary=dr_dict.get("feature_summary", {}),
                anomaly_score=dr_dict.get("anomaly_score", 0.0),
                similarity_score=dr_dict.get("similarity_score", 1.0),
                reputation_score=dr_dict.get("reputation_score", 1.0),
                explanation_codes=dr_dict.get("explanation_codes", []),
                detector_version=dr_dict.get("detector_version", "sentinel-v1"),
            )
            detections.append(detection)
            
        return detections

    def estimate_impact(self, updates: list[ModelUpdate], detections: list[DetectionResult], round_id: int) -> list[ImpactResult]:
        """
        Parse the cached impact estimates into canonical P4 ImpactResult objects.
        """
        results = self._round_cache.get(round_id, [])
        impacts = []
        
        for res in results:
            client_res = res.get("client_result", {})
            i_out = client_res.get("impact_output", {})
            update_id = i_out.get("update_id", "unknown")
            client_id = i_out.get("client_id", "unknown")
            
            # Map severity
            severity_str = i_out.get("impact_severity", "LOW")
            if severity_str in ("MINIMAL", "LOW"):
                impact_level = ImpactLevel.LOW
            elif severity_str == "MODERATE":
                impact_level = ImpactLevel.MEDIUM
            elif severity_str == "HIGH":
                impact_level = ImpactLevel.HIGH
            else:
                impact_level = ImpactLevel.CRITICAL

            breakdown = i_out.get("impact_breakdown", {})
            top_layers = i_out.get("top_impacted_layers", [])
            explanation_codes = []
            if severity_str:
                explanation_codes.append(f"severity:{severity_str}")
            for l in top_layers[:3]:
                if isinstance(l, dict) and "layer_name" in l:
                    explanation_codes.append(f"layer:{l['layer_name']}")
                
            impacts.append(ImpactResult(
                update_id=update_id,
                client_id=client_id,
                round_id=round_id,
                impact_score=float(i_out.get("impact_score", 0.0)),
                influence_estimate=float(i_out.get("threat_amplification", 0.0)),
                parameter_displacement=float(i_out.get("confidence_impact", 0.0)),
                aggregation_weight=1.0,
                impact_level=impact_level,
                explanation_codes=explanation_codes,
                impact_version=i_out.get("impact_version", "impact-v1"),
                impact_breakdown=breakdown,
                top_impacted_layers=top_layers,
                details=i_out,
            ))
            
        return impacts

    def decide_response(self, detections: list[DetectionResult], impacts: list[ImpactResult]) -> dict[str, ResponseAction]:
        """
        Return the mapping of update_id -> ResponseAction.
        """
        # The detect() method already extracted the final action into the DetectionResult
        return {d.update_id: d.action for d in detections}

    def check_recovery(self, evaluation: dict, detections: list[DetectionResult], impacts: list[ImpactResult], run_id: str, round_id: int, model_version: str, config: dict, val_metrics: dict | None = None, loss_spiked: bool = False) -> RecoveryResult:
        """
        Determine if recovery is needed based on P3's RecoveryTriggerEngine.
        """
        anomaly_scores = {d.client_id: d.anomaly_score for d in detections}
        
        decision = self.sentinel._recovery_trigger_engine.evaluate_recovery_need(
            anomaly_scores=anomaly_scores,
            val_metrics=val_metrics,
            loss_spiked=loss_spiked
        )
        
        excluded_clients = decision.quarantined_clients
        affected_updates = [d.update_id for d in detections if d.client_id in excluded_clients]
        
        recovery_status = RecoveryStatus.TRIGGERED if decision.is_triggered else RecoveryStatus.NOT_REQUIRED
        
        details = {
            "results_count": len(detections),
            "triggers": [decision.trigger_reason],
            "recovery_round": decision.recovery_round,
            "val_metrics": val_metrics,
            "loss_spiked": loss_spiked
        }

        return RecoveryResult(
            recovery_id=f"rec-{run_id}-{round_id}",
            run_id=run_id,
            round_id=round_id,
            trigger=decision.trigger_reason,
            affected_update_ids=affected_updates,
            excluded_client_ids=excluded_clients,
            previous_model_version=model_version,
            recovered_model_version="",
            before_accuracy=float(evaluation["accuracy"]) if evaluation and evaluation.get("accuracy") is not None else None,
            before_loss=float(evaluation["loss"]) if evaluation and evaluation.get("loss") is not None else None,
            recovery_status=recovery_status,
            recovery_version="recovery-v1",
            selected_action="ROLLBACK" if decision.is_triggered else None,
            details=details,
        )
