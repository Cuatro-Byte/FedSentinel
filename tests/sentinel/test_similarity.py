"""
tests/sentinel/test_similarity.py

Unit tests for core/sentinel/similarity.py
Owner: Person 3 — FedSentinel Detection, Impact & Recovery Intelligence
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from core.sentinel.feature_extractor import FeatureExtractor
from core.sentinel.sentinel import Sentinel
from core.sentinel.similarity import (
    SIMILARITY_ENGINE_VERSION,
    SCHEMA_VERSION,
    SimilarityEngine,
    SimilarityEngineError,
)

# ---------------------------------------------------------------------------
# Minimal ModelUpdate stub (reused from other tests)
# ---------------------------------------------------------------------------
class ModelUpdateStub:
    def __init__(self, update_id: str, client_id: str, round_id: int, parameters: dict[str, Any]):
        self.update_id = update_id
        self.client_id = client_id
        self.round_id = round_id
        self.parameters = parameters
        self.sample_count = 100
        self.training_epochs = 1
        self.learning_rate = 0.01
        self.local_loss = 0.5
        self.local_accuracy = 0.8

def _make_feature(update_id: str, client_id: str, params: np.ndarray) -> dict[str, Any]:
    update = ModelUpdateStub(update_id, client_id, 1, {"layer": params})
    return FeatureExtractor().extract(update)


@pytest.fixture
def engine() -> SimilarityEngine:
    return SimilarityEngine()


class TestSimilarityEngineCorrectness:
    """Mathematical correctness tests."""

    def test_identical_updates(self, engine: SimilarityEngine):
        """Two identical updates should have cos_sim = 1.0, dist = 0.0."""
        arr = np.array([1.0, 2.0, 3.0])
        f1 = _make_feature("u1", "c1", arr)
        f2 = _make_feature("u2", "c2", arr)
        
        results = engine.compute_similarities([f1, f2])
        assert len(results) == 2
        
        for r in results:
            assert abs(r["peer_similarity"]["average_similarity"] - 1.0) < 1e-6
            assert abs(r["peer_distance"]["average_distance"] - 0.0) < 1e-6
            assert abs(r["peer_similarity"]["median_similarity"] - 1.0) < 1e-6

    def test_orthogonal_updates(self, engine: SimilarityEngine):
        """Orthogonal vectors should have cos_sim = 0.0, dist = sqrt(2)."""
        f1 = _make_feature("u1", "c1", np.array([1.0, 0.0]))
        f2 = _make_feature("u2", "c2", np.array([0.0, 1.0]))
        
        results = engine.compute_similarities([f1, f2])
        assert len(results) == 2
        
        for r in results:
            assert abs(r["peer_similarity"]["average_similarity"] - 0.0) < 1e-6
            assert abs(r["peer_distance"]["average_distance"] - math.sqrt(2.0)) < 1e-6
            assert abs(r["consensus"]["consensus_score"] - 0.5) < 1e-6

    def test_opposite_updates(self, engine: SimilarityEngine):
        """Opposite vectors should have cos_sim = -1.0, dist = 2.0."""
        f1 = _make_feature("u1", "c1", np.array([1.0, 1.0]))
        f2 = _make_feature("u2", "c2", np.array([-1.0, -1.0]))
        
        results = engine.compute_similarities([f1, f2])
        
        for r in results:
            assert abs(r["peer_similarity"]["average_similarity"] - (-1.0)) < 1e-6
            assert abs(r["peer_distance"]["average_distance"] - 2.0) < 1e-6
            assert abs(r["consensus"]["consensus_score"] - 0.0) < 1e-6

    def test_cluster_center_correctness(self, engine: SimilarityEngine):
        f1 = _make_feature("u1", "c1", np.array([1.0, 0.0]))
        f2 = _make_feature("u2", "c2", np.array([0.0, 1.0]))
        f3 = _make_feature("u3", "c3", np.array([0.0, -1.0]))
        # Centroid is [1/3, 0]
        results = engine.compute_similarities([f1, f2, f3])
        
        # c1 distance to centroid
        c1_dist = float(np.linalg.norm(np.array([1.0, 0.0]) - np.array([1/3, 0.0])))
        assert abs(results[0]["peer_distance"]["cluster_distance"] - c1_dist) < 1e-6

    def test_median_similarity_correctness(self, engine: SimilarityEngine):
        f1 = _make_feature("u1", "c1", np.array([1.0, 0.0]))
        f2 = _make_feature("u2", "c2", np.array([2.0, 0.0]))
        f3 = _make_feature("u3", "c3", np.array([3.0, 0.0]))
        results = engine.compute_similarities([f1, f2, f3])
        # They are all co-linear, median should also be co-linear -> similarity 1.0
        for r in results:
            assert abs(r["peer_similarity"]["median_similarity"] - 1.0) < 1e-6

    def test_neighbor_count_correctness(self, engine: SimilarityEngine):
        f1 = _make_feature("u1", "c1", np.array([1.0, 0.0]))
        f2 = _make_feature("u2", "c2", np.array([1.0, 0.01])) # Very close
        f3 = _make_feature("u3", "c3", np.array([0.0, 1.0]))  # Far (0 similarity)
        
        results = engine.compute_similarities([f1, f2, f3])
        assert results[0]["consensus"]["neighbor_count"] == 1
        assert results[1]["consensus"]["neighbor_count"] == 1
        assert results[2]["consensus"]["neighbor_count"] == 0

    def test_similarity_ranking_correctness(self, engine: SimilarityEngine):
        f1 = _make_feature("u1", "c1", np.array([1.0, 1.0]))
        f2 = _make_feature("u2", "c2", np.array([1.0, 1.0])) # c1 and c2 are identical (rank 1 and 2)
        f3 = _make_feature("u3", "c3", np.array([-1.0, -1.0])) # opposite (rank 3)
        
        results = engine.compute_similarities([f1, f2, f3])
        r1 = next(r for r in results if r["client_id"] == "c1")
        r2 = next(r for r in results if r["client_id"] == "c2")
        r3 = next(r for r in results if r["client_id"] == "c3")
        
        assert r1["consensus"]["similarity_rank"] == 1
        assert r2["consensus"]["similarity_rank"] == 2
        assert r3["consensus"]["similarity_rank"] == 3

class TestSimilarityEngineEdgeCases:
    """Edge cases logic."""

    def test_single_client_round(self, engine: SimilarityEngine):
        f1 = _make_feature("u1", "c1", np.array([1.0, 2.0]))
        results = engine.compute_similarities([f1])
        assert len(results) == 1
        r = results[0]
        assert r["peer_similarity"]["average_similarity"] == 0.0
        assert r["peer_similarity"]["median_similarity"] == 1.0
        assert r["consensus"]["neighbor_count"] == 0
        assert r["consensus"]["similarity_rank"] == 1

    def test_zero_vectors(self, engine: SimilarityEngine):
        """Zero vectors shouldn't cause NaN due to EPSILON."""
        f1 = _make_feature("u1", "c1", np.zeros(10))
        f2 = _make_feature("u2", "c2", np.zeros(10))
        results = engine.compute_similarities([f1, f2])
        for r in results:
            assert math.isfinite(r["peer_similarity"]["average_similarity"])

    def test_constant_vectors(self, engine: SimilarityEngine):
        f1 = _make_feature("u1", "c1", np.ones(10))
        f2 = _make_feature("u2", "c2", np.ones(10) * 5.0)
        results = engine.compute_similarities([f1, f2])
        # Cosine similarity between c*1 and k*1 is 1.0
        assert abs(results[0]["peer_similarity"]["average_similarity"] - 1.0) < 1e-6

    def test_sparse_vectors(self, engine: SimilarityEngine):
        arr1 = np.zeros(100); arr1[0] = 1.0
        arr2 = np.zeros(100); arr2[1] = 1.0
        f1 = _make_feature("u1", "c1", arr1)
        f2 = _make_feature("u2", "c2", arr2)
        results = engine.compute_similarities([f1, f2])
        assert abs(results[0]["peer_similarity"]["average_similarity"] - 0.0) < 1e-6

    def test_empty_round(self, engine: SimilarityEngine):
        results = engine.compute_similarities([])
        assert results == []

    def test_duplicate_client_ids(self, engine: SimilarityEngine):
        f1 = _make_feature("u1", "c1", np.array([1.0]))
        f2 = _make_feature("u2", "c1", np.array([2.0]))
        with pytest.raises(SimilarityEngineError):
            engine.compute_similarities([f1, f2])

    def test_mixed_tensor_sizes(self, engine: SimilarityEngine):
        f1 = _make_feature("u1", "c1", np.array([1.0, 2.0]))
        f2 = _make_feature("u2", "c2", np.array([1.0, 2.0, 3.0]))
        with pytest.raises(SimilarityEngineError):
            engine.compute_similarities([f1, f2])

    def test_missing_flat_parameters(self, engine: SimilarityEngine):
        f1 = _make_feature("u1", "c1", np.array([1.0, 2.0]))
        del f1["flat_parameters"]
        with pytest.raises(SimilarityEngineError):
            engine.compute_similarities([f1])


class TestSimilarityEngineQuality:
    """Ensure compliance with data contracts."""

    def test_schema_versions(self, engine: SimilarityEngine):
        f1 = _make_feature("u1", "c1", np.array([1.0]))
        f2 = _make_feature("u2", "c2", np.array([2.0]))
        results = engine.compute_similarities([f1, f2])
        assert results[0]["schema_version"] == SCHEMA_VERSION
        assert results[0]["similarity_version"] == SIMILARITY_ENGINE_VERSION

    def test_no_numpy_types_leak(self, engine: SimilarityEngine):
        f1 = _make_feature("u1", "c1", np.array([1.0, 2.0]))
        f2 = _make_feature("u2", "c2", np.array([3.0, 4.0]))
        results = engine.compute_similarities([f1, f2])
        
        def check_no_numpy(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    check_no_numpy(v)
            elif isinstance(obj, list):
                for v in obj:
                    check_no_numpy(v)
            else:
                assert not isinstance(obj, np.ndarray), "Numpy array leaked!"
                assert not isinstance(obj, np.generic), f"Numpy scalar leaked: {type(obj)}"

        check_no_numpy(results)

    def test_via_sentinel(self):
        sentinel = Sentinel()
        f1 = _make_feature("u1", "c1", np.array([1.0]))
        f2 = _make_feature("u2", "c2", np.array([2.0]))
        results = sentinel.compute_similarity([f1, f2])
        assert len(results) == 2


class TestSimilarityEnginePerformance:
    def test_large_tensors(self, engine: SimilarityEngine):
        """20 clients x 10M parameters should run efficiently."""
        import time
        rng = np.random.default_rng(42)
        features = []
        for i in range(20):
            # To save actual memory in test environment, we simulate a smaller matrix
            # But here we verify performance with 1M params to fit typical test RAM limits easily.
            arr = rng.normal(0, 1, 1_000_000)
            features.append(_make_feature(f"u{i}", f"c{i}", arr))
        
        t0 = time.perf_counter()
        results = engine.compute_similarities(features)
        t1 = time.perf_counter()
        
        assert len(results) == 20
        # Time should be minimal due to vectorized operations
        assert (t1 - t0) < 5.0  # Allow some overhead
