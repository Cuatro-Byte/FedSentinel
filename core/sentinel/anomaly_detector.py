"""
core/sentinel/anomaly_detector.py

FedSentinel — Anomaly Detection Engine
Owner: Person 3 — FedSentinel Detection, Impact & Recovery Intelligence

This module computes an anomaly score [0,1] for a single client update
relative to its descriptive statistics and peer similarities.

Contract reference: FEDSENTINEL_NEW_TEAM_ENGINEERING_CONTRACT_v2.md
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

ANOMALY_ENGINE_VERSION: str = "anomaly-v1"
SCHEMA_VERSION: str = "schema-v1"
EPSILON: float = 1e-12

logger = logging.getLogger(__name__)


class AnomalyDetectorError(Exception):
    """
    Raised when the Anomaly Detector cannot safely process the inputs.
    (e.g., missing fields, NaN/Inf values).
    """


class AnomalyDetector:
    """
    Computes a composite anomaly score for a single client update.

    This class is stateless. It expects the outputs of the FeatureExtractor,
    StatisticsEngine, and SimilarityEngine for a single client.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(
            self.__class__.__module__ + "." + self.__class__.__name__
        )
        self._logger.debug(
            "AnomalyDetector initialised (version=%s schema=%s)",
            ANOMALY_ENGINE_VERSION,
            SCHEMA_VERSION,
        )
        # Default prototype weights as per the prompt
        self.weights = {
            "magnitude": 0.30,
            "distribution": 0.25,
            "similarity": 0.30,
            "sparsity": 0.15
        }

    def compute_anomaly(self, feature: dict[str, Any], statistics: dict[str, Any], similarity: dict[str, Any]) -> dict[str, Any]:
        """
        Compute the anomaly score for a single client.

        Parameters
        ----------
        feature:
            The feature dictionary produced by Phase 1.
        statistics:
            The statistics dictionary produced by Phase 2.
        similarity:
            The peer similarity dictionary produced by Phase 3.

        Returns
        -------
        dict[str, Any]
            The anomaly dictionary matching the strict Output Contract.

        Raises
        ------
        AnomalyDetectorError
            If inputs are invalid or incomplete.
        """
        # Validate inputs
        if not feature or not statistics or not similarity:
            raise AnomalyDetectorError("Missing input dictionaries for anomaly detection.")

        update_id = feature.get("update_id")
        client_id = feature.get("client_id")
        round_id = feature.get("round_id")

        if not update_id or not client_id:
            raise AnomalyDetectorError("Missing update_id or client_id in feature dictionary.")

        global_stats = statistics.get("global_statistics")
        layer_stats = statistics.get("layer_statistics")
        if not global_stats or not layer_stats:
            raise AnomalyDetectorError(f"Missing statistics data for update {update_id}")

        peer_sim = similarity.get("peer_similarity")
        consensus = similarity.get("consensus")
        peer_dist = similarity.get("peer_distance")
        if not peer_sim or not consensus or not peer_dist:
            raise AnomalyDetectorError(f"Missing similarity data for update {update_id}")

        # Compute Independent Signals
        try:
            magnitude_anomaly = self._compute_magnitude_anomaly(global_stats)
            distribution_anomaly = self._compute_distribution_anomaly(global_stats)
            similarity_anomaly = self._compute_similarity_anomaly(peer_sim, consensus, peer_dist)
            sparsity_anomaly = self._compute_sparsity_anomaly(global_stats)
            
            layer_anomaly_scores = self._compute_layer_anomaly_scores(layer_stats)
            
        except Exception as e:
            raise AnomalyDetectorError(f"Failed to compute anomaly signals: {e}") from e

        # Validate bounds [0, 1]
        magnitude_anomaly = self._safe_clip(magnitude_anomaly)
        distribution_anomaly = self._safe_clip(distribution_anomaly)
        similarity_anomaly = self._safe_clip(similarity_anomaly)
        sparsity_anomaly = self._safe_clip(sparsity_anomaly)

        # Composite Score
        raw_signals = {
            "magnitude_anomaly": magnitude_anomaly,
            "distribution_anomaly": distribution_anomaly,
            "similarity_anomaly": similarity_anomaly,
            "sparsity_anomaly": sparsity_anomaly
        }
        
        weighted_sum = (
            magnitude_anomaly * self.weights["magnitude"] +
            distribution_anomaly * self.weights["distribution"] +
            similarity_anomaly * self.weights["similarity"] +
            sparsity_anomaly * self.weights["sparsity"]
        )
        
        final_anomaly_score = self._safe_clip(weighted_sum)
        
        # Ensure JSON-native types
        layer_scores_native = {str(k): float(v) for k, v in layer_anomaly_scores.items()}

        result = {
            "update_id": update_id,
            "client_id": client_id,
            "round_id": round_id,
            "schema_version": SCHEMA_VERSION,
            "anomaly_version": ANOMALY_ENGINE_VERSION,
            "anomaly_score": float(final_anomaly_score),
            "signal_breakdown": {
                "magnitude_anomaly": float(magnitude_anomaly),
                "distribution_anomaly": float(distribution_anomaly),
                "similarity_anomaly": float(similarity_anomaly),
                "sparsity_anomaly": float(sparsity_anomaly)
            },
            "layer_anomaly_scores": layer_scores_native,
            "normalization": {
                "minimum_signal": float(min(raw_signals.values())),
                "maximum_signal": float(max(raw_signals.values())),
                "weighted_sum": float(weighted_sum)
            }
        }

        self._logger.info("Computed anomaly score %f for client %s", final_anomaly_score, client_id)
        return result

    def _safe_clip(self, val: float) -> float:
        if not math.isfinite(val):
            return 1.0  # Inf/NaN anomalies default to maximum anomaly
        return float(np.clip(val, 0.0, 1.0))

    def _compute_magnitude_anomaly(self, global_stats: dict[str, Any]) -> float:
        """
        Derive magnitude anomaly from L2 norm, update magnitude, and RMS deviation.
        A heuristic z-score replacement bounded to [0, 1].
        """
        magnitude = float(global_stats.get("update_magnitude", 0.0))
        rms = float(global_stats.get("rms_deviation", 0.0))
        
        # Assume extremely high or extremely low (zero) magnitudes are anomalous.
        # Heuristic mapping: map magnitude to [0,1].
        # Using a simple scaling: normal magnitude ~1.0. 
        mag_score = abs(magnitude - 1.0) / (abs(magnitude - 1.0) + 1.0 + EPSILON)
        rms_score = rms / (rms + 1.0 + EPSILON)
        
        return (mag_score + rms_score) / 2.0

    def _compute_distribution_anomaly(self, global_stats: dict[str, Any]) -> float:
        """
        Derive distribution anomaly from Skewness, Kurtosis, MAD, and IQR.
        """
        skewness = abs(float(global_stats.get("skewness", 0.0)))
        kurtosis = abs(float(global_stats.get("kurtosis", 0.0)))
        mad = float(global_stats.get("median_absolute_deviation", 0.0))
        iqr = float(global_stats.get("interquartile_range", 0.0))
        
        skew_score = skewness / (skewness + 5.0 + EPSILON)
        kurt_score = kurtosis / (kurtosis + 10.0 + EPSILON)
        mad_score = mad / (mad + 1.0 + EPSILON)
        iqr_score = iqr / (iqr + 1.0 + EPSILON)
        
        return (skew_score + kurt_score + mad_score + iqr_score) / 4.0

    def _compute_similarity_anomaly(self, peer_sim: dict[str, Any], consensus: dict[str, Any], peer_dist: dict[str, Any]) -> float:
        """
        Derive similarity anomaly from consensus_score, cluster_distance, and avg_similarity.
        Low similarity -> High anomaly. High consensus -> Low anomaly.
        """
        avg_sim = float(peer_sim.get("average_similarity", 0.0))
        cons_score = float(consensus.get("consensus_score", 1.0))
        cluster_dist = float(peer_dist.get("normalized_cluster_distance", 0.0))
        
        # cons_score is [0, 1], where 1 is strongly agrees. 
        # Anomaly is inverse of consensus.
        cons_anomaly = 1.0 - cons_score
        
        # avg_sim is [-1, 1]. Map to [0, 1] anomaly.
        # 1 -> 0 anomaly, -1 -> 1 anomaly.
        sim_anomaly = (1.0 - avg_sim) / 2.0
        
        # cluster_dist is >= 0. High dist -> high anomaly.
        dist_anomaly = cluster_dist / (cluster_dist + 1.0 + EPSILON)
        
        return (cons_anomaly + sim_anomaly + dist_anomaly) / 3.0

    def _compute_sparsity_anomaly(self, global_stats: dict[str, Any]) -> float:
        """
        Derive sparsity anomaly from parameter_sparsity and non_zero_ratio.
        """
        sparsity = float(global_stats.get("parameter_sparsity", 0.0))
        nzr = float(global_stats.get("non_zero_ratio", 1.0))
        
        # Assume normal dense neural net updates have sparsity near 0.0.
        # High sparsity is anomalous.
        # Conversely, if a client is completely zero, sparsity is 1.0 -> anomaly 1.0.
        # If it's expected to be sparse, we'd need a baseline, but without one we map directly.
        sparsity_anomaly = sparsity
        
        return sparsity_anomaly

    def _compute_layer_anomaly_scores(self, layer_stats: dict[str, dict[str, Any]]) -> dict[str, float]:
        """
        Derive anomaly score [0, 1] per layer.
        """
        scores = {}
        for layer_name, stats in layer_stats.items():
            l2 = float(stats.get("layer_l2_norm", 0.0))
            var = float(stats.get("layer_variance", 0.0))
            
            l2_score = l2 / (l2 + 1.0 + EPSILON)
            var_score = var / (var + 1.0 + EPSILON)
            
            layer_anomaly = (l2_score + var_score) / 2.0
            scores[layer_name] = self._safe_clip(layer_anomaly)
            
        return scores
