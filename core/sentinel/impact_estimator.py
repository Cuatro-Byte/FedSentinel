"""
core/sentinel/impact_estimator.py

FedSentinel — Impact Estimation Engine
Owner: Person 3 — FedSentinel Detection, Impact & Recovery Intelligence

Estimates potential global model impact of a client update.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

IMPACT_ENGINE_VERSION: str = "impact-v1"
SCHEMA_VERSION: str = "schema-v1"
EPSILON: float = 1e-12

MODEL_DRIFT_WEIGHT: float = 0.35
LAYER_IMPACT_WEIGHT: float = 0.25
THREAT_AMPLIFICATION_WEIGHT: float = 0.20
HISTORICAL_DAMAGE_WEIGHT: float = 0.10
CONFIDENCE_WEIGHT: float = 0.10

MINIMAL_THRESHOLD: float = 0.20
LOW_THRESHOLD: float = 0.40
MODERATE_THRESHOLD: float = 0.60
HIGH_THRESHOLD: float = 0.80
SEVERE_THRESHOLD: float = 1.00


class ImpactEstimatorError(Exception):
    pass


class ImpactEstimator:
    def __init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__module__ + "." + self.__class__.__name__)
        self._logger.debug("ImpactEstimator initialised (version=%s)", IMPACT_ENGINE_VERSION)
        
        # Verify weights sum to 1.0
        total_weight = MODEL_DRIFT_WEIGHT + LAYER_IMPACT_WEIGHT + THREAT_AMPLIFICATION_WEIGHT + HISTORICAL_DAMAGE_WEIGHT + CONFIDENCE_WEIGHT
        if abs(total_weight - 1.0) > 1e-9:
            raise ValueError(f"Impact weights must sum to 1.0, got {total_weight}")

    def compute_impact(self, 
                       feature: dict[str, Any], 
                       statistics: dict[str, Any], 
                       similarity: dict[str, Any], 
                       anomaly: dict[str, Any], 
                       reputation: dict[str, Any],
                       threat: dict[str, Any]) -> dict[str, Any]:
        """
        Compute the potential impact of a client update.
        """
        if feature is None or statistics is None or similarity is None or anomaly is None or reputation is None or threat is None:
            raise ImpactEstimatorError("Missing input signals for impact estimation.")

        try:
            update_id = feature.get("update_id")
            client_id = feature.get("client_id")
            round_id = feature.get("round_id")
            
            if not update_id or not client_id or round_id is None:
                raise ValueError("Missing update_id, client_id, or round_id.")

            # Model Drift Impact
            global_stats = statistics.get("global_statistics", {})
            l2_norm = float(global_stats.get("l2_norm", 0.0))
            rms = float(global_stats.get("rms", 0.0))
            energy = float(global_stats.get("energy", 0.0))
            model_drift = self._clip(1.0 - (1.0 / (1.0 + l2_norm + rms + (energy / 100.0))))

            # Layer Impact
            layer_stats = statistics.get("layer_statistics", {})
            layer_shapes = feature.get("layer_shapes", {})
            if not layer_shapes:
                raise ImpactEstimatorError("Missing layer_shapes in feature input.")
                
            layer_impact_scores = {}
            for layer_name in layer_shapes.keys():
                ls = layer_stats.get(layer_name, {})
                l_rms = float(ls.get("rms", 0.0))
                l_var = float(ls.get("variance", 0.0))
                l_mag = float(ls.get("max_abs", 0.0))
                score = self._clip(1.0 - (1.0 / (1.0 + l_rms + l_var + l_mag)))
                layer_impact_scores[layer_name] = float(score)

            # Average layer impact
            avg_layer_impact = float(np.mean(list(layer_impact_scores.values()))) if layer_impact_scores else 0.0

            # Threat Amplification
            threat_score = float(threat.get("threat_score", 0.0))
            confidence_score = float(threat.get("confidence_score", 1.0))
            threat_stability = confidence_score
            threat_amplification = self._clip(threat_score * (1.0 + (1.0 - threat_stability)))

            # Historical Damage Risk
            rep_counts = reputation.get("behavior_counts", {})
            suspicious = int(rep_counts.get("suspicious_count", 0))
            malicious = int(rep_counts.get("malicious_count", 0))
            quarantine = int(rep_counts.get("quarantine_count", 0))
            historical_damage_risk = self._clip((suspicious + malicious * 2 + quarantine * 3) / 10.0)

            # Confidence Impact
            rep_metrics = reputation.get("reputation_metrics", {})
            rounds_seen = int(rep_metrics.get("rounds_seen", 1))
            consistency = float(rep_metrics.get("consistency_score", 1.0))
            conf_impact = self._clip((min(rounds_seen / 10.0, 1.0) * 0.5) + (consistency * 0.5))

            # Composite Impact
            raw_impact = (
                model_drift * MODEL_DRIFT_WEIGHT +
                avg_layer_impact * LAYER_IMPACT_WEIGHT +
                threat_amplification * THREAT_AMPLIFICATION_WEIGHT +
                historical_damage_risk * HISTORICAL_DAMAGE_WEIGHT +
                conf_impact * CONFIDENCE_WEIGHT
            )
            impact_score = self._clip(raw_impact)

            # Map Severity
            impact_severity = self._map_severity(impact_score)

            # Breakdown percentages (must sum to exactly 100.0)
            total_raw = (model_drift * MODEL_DRIFT_WEIGHT +
                         avg_layer_impact * LAYER_IMPACT_WEIGHT +
                         threat_amplification * THREAT_AMPLIFICATION_WEIGHT +
                         historical_damage_risk * HISTORICAL_DAMAGE_WEIGHT +
                         conf_impact * CONFIDENCE_WEIGHT + EPSILON)
            
            p_model = (model_drift * MODEL_DRIFT_WEIGHT) / total_raw * 100.0
            p_layer = (avg_layer_impact * LAYER_IMPACT_WEIGHT) / total_raw * 100.0
            p_threat = (threat_amplification * THREAT_AMPLIFICATION_WEIGHT) / total_raw * 100.0
            p_hist = (historical_damage_risk * HISTORICAL_DAMAGE_WEIGHT) / total_raw * 100.0
            p_conf = (conf_impact * CONFIDENCE_WEIGHT) / total_raw * 100.0

            # Round and correct for exactly 100.0
            p_list = [
                ("model_drift", np.round(p_model, 2)),
                ("layer_impact", np.round(p_layer, 2)),
                ("threat_amplification", np.round(p_threat, 2)),
                ("historical_damage", np.round(p_hist, 2)),
                ("confidence", np.round(p_conf, 2))
            ]
            
            # Fix rounding drift by adding difference to the largest component
            current_sum = sum(v for _, v in p_list)
            diff = 100.0 - current_sum
            if abs(diff) > 1e-9:
                p_list.sort(key=lambda x: x[1], reverse=True)
                p_list[0] = (p_list[0][0], p_list[0][1] + diff)
                
            breakdown = {k: float(np.round(v, 2)) for k, v in p_list}

            # Top impacted layers (sort descending by score, then layer_name)
            sorted_layers = sorted(
                layer_impact_scores.items(),
                key=lambda x: (-x[1], x[0])
            )
            top_5 = sorted_layers[:5]
            
            total_top_score = sum(score for _, score in top_5) + EPSILON
            top_impacted_layers = []
            
            top_p_list = []
            for name, score in top_5:
                top_p_list.append({
                    "layer_name": name,
                    "impact_score": float(score),
                    "raw_pct": (score / total_raw) * 100.0 if total_raw > EPSILON else 0.0,
                    "contribution_percentage": (score / total_top_score) * 100.0
                })
            
            # Correct top 5 contributions to exactly 100.0
            if top_p_list:
                for i in range(len(top_p_list)):
                    top_p_list[i]["contribution_percentage"] = np.round(top_p_list[i]["contribution_percentage"], 2)
                    
                t_sum = sum(x["contribution_percentage"] for x in top_p_list)
                t_diff = 100.0 - t_sum
                if abs(t_diff) > 1e-9:
                    top_p_list.sort(key=lambda x: (-x["contribution_percentage"], x["layer_name"]))
                    top_p_list[0]["contribution_percentage"] += t_diff
                    
                # Re-sort to original (-score, name)
                top_p_list.sort(key=lambda x: (-x["impact_score"], x["layer_name"]))
                
                for item in top_p_list:
                    top_impacted_layers.append({
                        "layer_name": item["layer_name"],
                        "impact_score": item["impact_score"],
                        "contribution_percentage": float(np.round(item["contribution_percentage"], 2))
                    })

            return {
                "update_id": update_id,
                "client_id": client_id,
                "round_id": round_id,
                "schema_version": SCHEMA_VERSION,
                "impact_version": IMPACT_ENGINE_VERSION,
                "impact_score": float(impact_score),
                "impact_severity": impact_severity,
                "confidence_impact": float(conf_impact),
                "historical_damage_risk": float(historical_damage_risk),
                "threat_amplification": float(threat_amplification),
                "impact_breakdown": breakdown,
                "top_impacted_layers": top_impacted_layers,
                "layer_impact_scores": layer_impact_scores
            }
            
        except Exception as e:
            raise ImpactEstimatorError(f"Failed to compute impact: {e}") from e

    def _clip(self, val: float) -> float:
        if not math.isfinite(val):
            return 1.0
        return float(np.clip(val, 0.0, 1.0))
        
    def _map_severity(self, score: float) -> str:
        if score <= MINIMAL_THRESHOLD:
            return "MINIMAL"
        elif score <= LOW_THRESHOLD:
            return "LOW"
        elif score <= MODERATE_THRESHOLD:
            return "MODERATE"
        elif score <= HIGH_THRESHOLD:
            return "HIGH"
        else:
            return "SEVERE"
