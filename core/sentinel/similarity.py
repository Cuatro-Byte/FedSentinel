"""
core/sentinel/similarity.py

FedSentinel — Peer Similarity Engine
Owner: Person 3 — FedSentinel Detection, Impact & Recovery Intelligence

This module computes pairwise similarities and distances between client
updates in a given federated round. It provides descriptive statistics
that serve as the foundation for anomaly detection in later phases.

Contract reference: FEDSENTINEL_NEW_TEAM_ENGINEERING_CONTRACT_v2.md
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

SIMILARITY_ENGINE_VERSION: str = "similarity-v1"
SCHEMA_VERSION: str = "schema-v1"
EPSILON: float = 1e-12
SIMILARITY_THRESHOLD: float = 0.80

logger = logging.getLogger(__name__)


class SimilarityEngineError(Exception):
    """
    Raised when the Similarity Engine cannot safely process the round updates.
    (e.g., duplicate client IDs, varying tensor sizes, corrupted features).
    """


class SimilarityEngine:
    """
    Computes peer similarities for a set of client updates in a single round.

    This class is stateless. It expects a list of feature dictionaries
    produced by the FeatureExtractor.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(
            self.__class__.__module__ + "." + self.__class__.__name__
        )
        self._logger.debug(
            "SimilarityEngine initialised (version=%s schema=%s)",
            SIMILARITY_ENGINE_VERSION,
            SCHEMA_VERSION,
        )

    def compute_similarities(self, features_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Compute similarity metrics for a list of client updates.

        Parameters
        ----------
        features_list:
            A list of feature dictionaries, each conforming to Phase 1 schema.
            Must belong to the same round.

        Returns
        -------
        list[dict[str, Any]]
            A list of dictionaries, one per client, containing similarity,
            distance, and consensus metrics. Exactly matches the Output Contract.
            Does NOT leak raw numpy arrays.

        Raises
        ------
        SimilarityEngineError
            If updates cannot be safely compared.
        """
        if not features_list:
            self._logger.warning("Empty features_list provided to SimilarityEngine.")
            return []

        N = len(features_list)
        
        # 1. Validate inputs and extract raw arrays safely
        client_ids = []
        arrays = []
        expected_size = None
        
        for f in features_list:
            cid = f.get("client_id")
            if not cid:
                raise SimilarityEngineError(f"Missing client_id in feature dict: {f.get('update_id')}")
            client_ids.append(cid)
            
            flat = f.get("flat_parameters")
            if flat is None or not isinstance(flat, np.ndarray):
                raise SimilarityEngineError(f"Missing or invalid 'flat_parameters' for client {cid}")
            
            if expected_size is None:
                expected_size = flat.size
            elif flat.size != expected_size:
                raise SimilarityEngineError(
                    f"Tensor size mismatch: expected {expected_size}, got {flat.size} for client {cid}"
                )
            
            arrays.append(flat)

        if len(set(client_ids)) != N:
            raise SimilarityEngineError("Duplicate client IDs detected in the round updates.")

        P = np.stack(arrays)  # Shape: (N, D)
        
        # Determine base attributes from the first item (assuming same round)
        round_id = features_list[0].get("round_id")

        # 2. Handle single client edge case
        if N == 1:
            return [self._build_single_client_output(features_list[0])]

        # 3. Vectorized Math Computations
        
        # Norms and normalized vectors
        L2 = np.linalg.norm(P, axis=1, keepdims=True)
        P_norm = P / np.maximum(L2, EPSILON)

        # Pairwise Cosine Similarity
        # C[i, j] = cosine similarity between i and j
        C = P_norm @ P_norm.T
        C = np.clip(C, -1.0, 1.0)
        
        # Normalized Euclidean Distance
        # dist^2 = ||u - v||^2 = ||u||^2 + ||v||^2 - 2(u.v) = 2 - 2*cos
        # Using normalized vectors, so ||u|| = 1
        D = np.sqrt(np.clip(2.0 - 2.0 * C, 0.0, None))

        # Ignore self-comparison for aggregate stats
        np.fill_diagonal(C, np.nan)
        np.fill_diagonal(D, np.nan)

        # 4. Global properties (Median Update, Centroid, Median Magnitude)
        median_update = np.median(P, axis=0)
        median_update_norm_val = np.maximum(np.linalg.norm(median_update), EPSILON)
        median_normalized = median_update / median_update_norm_val
        
        centroid = np.mean(P, axis=0)
        centroid_norm_val = np.maximum(np.linalg.norm(centroid), EPSILON)
        centroid_normalized = centroid / centroid_norm_val

        median_L2 = float(np.median(L2))

        # 5. Extract per-client statistics
        results = []
        avg_sims = []
        
        for i, f in enumerate(features_list):
            uid = f.get("update_id", "unknown")
            cid = client_ids[i]

            # Peer Similarity Stats
            c_row = C[i]
            avg_sim = float(np.nanmean(c_row))
            min_sim = float(np.nanmin(c_row))
            max_sim = float(np.nanmax(c_row))
            std_sim = float(np.nanstd(c_row))
            avg_sims.append(avg_sim)

            med_sim = float(np.dot(P_norm[i], median_normalized))

            # Peer Distance Stats
            d_row = D[i]
            avg_dist = float(np.nanmean(d_row))
            min_dist = float(np.nanmin(d_row))
            max_dist = float(np.nanmax(d_row))

            c_dist = float(np.linalg.norm(P[i] - centroid))
            nc_dist = float(np.linalg.norm(P_norm[i] - centroid_normalized))

            l2_val = float(L2[i, 0])
            mag_rel_dev = float(abs(l2_val - median_L2) / (median_L2 + EPSILON))

            # Consensus Stats
            neighbors = np.sum(c_row >= SIMILARITY_THRESHOLD)
            neighbor_count = int(neighbors)
            neighbor_ratio = float(neighbors / (N - 1))
            
            # Map [-1, 1] cosine similarity to [0, 1] for a descriptive consensus score
            consensus_score = float(np.clip((avg_sim + 1.0) / 2.0, 0.0, 1.0))

            # Assure Python native types for JSON serialization
            def _clean_float(v: float) -> float:
                if not math.isfinite(v):
                    return 0.0  # Safe fallback for all-zero comparisons
                return float(v)

            results.append({
                "update_id": uid,
                "client_id": cid,
                "round_id": round_id,
                "schema_version": SCHEMA_VERSION,
                "similarity_version": SIMILARITY_ENGINE_VERSION,
                "peer_similarity": {
                    "average_similarity": _clean_float(avg_sim),
                    "median_similarity": _clean_float(med_sim),
                    "minimum_similarity": _clean_float(min_sim),
                    "maximum_similarity": _clean_float(max_sim),
                    "similarity_std": _clean_float(std_sim)
                },
                "peer_distance": {
                    "average_distance": _clean_float(avg_dist),
                    "minimum_distance": _clean_float(min_dist),
                    "maximum_distance": _clean_float(max_dist),
                    "cluster_distance": _clean_float(c_dist),
                    "normalized_cluster_distance": _clean_float(nc_dist),
                    "magnitude_relative_deviation": _clean_float(mag_rel_dev)
                },
                "consensus": {
                    "consensus_score": _clean_float(consensus_score),
                    "neighbor_count": neighbor_count,
                    "neighbor_ratio": _clean_float(neighbor_ratio),
                    # similarity_rank populated below
                    "similarity_rank": 0
                }
            })

        # 6. Similarity Ranking
        # Higher avg_sim -> better rank (1 is highest). Tie break on client_id ascending.
        # We sort by (-avg_sim, client_id)
        ranking_order = sorted(
            range(N),
            key=lambda idx: (-results[idx]["peer_similarity"]["average_similarity"], results[idx]["client_id"])
        )
        
        for rank_0_idx, i in enumerate(ranking_order):
            results[i]["consensus"]["similarity_rank"] = rank_0_idx + 1

        self._logger.info("Computed peer similarities for %d clients in round %s", N, round_id)
        return results

    def _build_single_client_output(self, f: dict[str, Any]) -> dict[str, Any]:
        """Edge case: Round has exactly one client."""
        return {
            "update_id": f.get("update_id", "unknown"),
            "client_id": f.get("client_id", "unknown"),
            "round_id": f.get("round_id"),
            "schema_version": SCHEMA_VERSION,
            "similarity_version": SIMILARITY_ENGINE_VERSION,
            "peer_similarity": {
                "average_similarity": 0.0,
                "median_similarity": 1.0,
                "minimum_similarity": 0.0,
                "maximum_similarity": 0.0,
                "similarity_std": 0.0
            },
            "peer_distance": {
                "average_distance": 0.0,
                "minimum_distance": 0.0,
                "maximum_distance": 0.0,
                "cluster_distance": 0.0,
                "normalized_cluster_distance": 0.0,
                "magnitude_relative_deviation": 0.0
            },
            "consensus": {
                "consensus_score": 1.0,
                "neighbor_count": 0,
                "neighbor_ratio": 0.0,
                "similarity_rank": 1
            }
        }
