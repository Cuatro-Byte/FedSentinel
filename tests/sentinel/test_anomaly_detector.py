"""
tests/sentinel/test_anomaly_detector.py

Unit tests for core/sentinel/anomaly_detector.py
Owner: Person 3 — FedSentinel Detection, Impact & Recovery Intelligence
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from core.sentinel.anomaly_detector import (
    ANOMALY_ENGINE_VERSION,
    SCHEMA_VERSION,
    AnomalyDetector,
    AnomalyDetectorError,
)

@pytest.fixture
def detector() -> AnomalyDetector:
    return AnomalyDetector()

def make_dummy_feature(uid: str, cid: str) -> dict[str, Any]:
    return {
        "update_id": uid,
        "client_id": cid,
        "round_id": 1,
        "schema_version": SCHEMA_VERSION
    }

def make_dummy_stats(mag: float=1.0, rms: float=0.1, skew: float=0.0, kurt: float=0.0,
                     mad: float=0.1, iqr: float=0.2, spar: float=0.0, nzr: float=1.0,
                     layer_l2: float=1.0, layer_var: float=0.1) -> dict[str, Any]:
    return {
        "global_statistics": {
            "update_magnitude": mag,
            "rms_deviation": rms,
            "skewness": skew,
            "kurtosis": kurt,
            "median_absolute_deviation": mad,
            "interquartile_range": iqr,
            "parameter_sparsity": spar,
            "non_zero_ratio": nzr
        },
        "layer_statistics": {
            "layer_1": {
                "layer_l2_norm": layer_l2,
                "layer_variance": layer_var
            }
        }
    }

def make_dummy_sim(avg_sim: float=1.0, c_dist: float=0.0, cons: float=1.0) -> dict[str, Any]:
    return {
        "peer_similarity": {
            "average_similarity": avg_sim,
        },
        "peer_distance": {
            "normalized_cluster_distance": c_dist,
        },
        "consensus": {
            "consensus_score": cons,
        }
    }

class TestAnomalyDetectorCorrectness:
    """Mathematical correctness tests."""

    def test_normal_magnitude(self, detector: AnomalyDetector):
        feat = make_dummy_feature("u1", "c1")
        # normal magnitude is near 1.0, low RMS
        stats = make_dummy_stats(mag=1.0, rms=0.0)
        sim = make_dummy_sim()
        res = detector.compute_anomaly(feat, stats, sim)
        mag_anom = res["signal_breakdown"]["magnitude_anomaly"]
        # Expected: abs(1-1) = 0 -> mag_score 0, rms_score 0 -> total 0
        assert mag_anom < 0.01

    def test_high_magnitude(self, detector: AnomalyDetector):
        feat = make_dummy_feature("u1", "c1")
        stats = make_dummy_stats(mag=100.0, rms=10.0)
        sim = make_dummy_sim()
        res = detector.compute_anomaly(feat, stats, sim)
        mag_anom = res["signal_breakdown"]["magnitude_anomaly"]
        assert mag_anom > 0.9

    def test_low_magnitude(self, detector: AnomalyDetector):
        feat = make_dummy_feature("u1", "c1")
        stats = make_dummy_stats(mag=0.0, rms=0.0)
        sim = make_dummy_sim()
        res = detector.compute_anomaly(feat, stats, sim)
        mag_anom = res["signal_breakdown"]["magnitude_anomaly"]
        # mag_score = 1 / (1 + 1) = 0.5, rms_score = 0 -> 0.25
        assert abs(mag_anom - 0.25) < 0.01

    def test_high_skewness_kurtosis(self, detector: AnomalyDetector):
        feat = make_dummy_feature("u1", "c1")
        stats = make_dummy_stats(skew=50.0, kurt=100.0)
        sim = make_dummy_sim()
        res = detector.compute_anomaly(feat, stats, sim)
        dist_anom = res["signal_breakdown"]["distribution_anomaly"]
        assert dist_anom > 0.4  # It should be significantly higher than 0

    def test_high_mad_iqr(self, detector: AnomalyDetector):
        feat = make_dummy_feature("u1", "c1")
        stats = make_dummy_stats(mad=100.0, iqr=100.0)
        sim = make_dummy_sim()
        res = detector.compute_anomaly(feat, stats, sim)
        dist_anom = res["signal_breakdown"]["distribution_anomaly"]
        assert dist_anom > 0.4

    def test_high_similarity(self, detector: AnomalyDetector):
        feat = make_dummy_feature("u1", "c1")
        stats = make_dummy_stats()
        sim = make_dummy_sim(avg_sim=1.0, c_dist=0.0, cons=1.0)
        res = detector.compute_anomaly(feat, stats, sim)
        sim_anom = res["signal_breakdown"]["similarity_anomaly"]
        assert sim_anom < 0.01

    def test_low_similarity(self, detector: AnomalyDetector):
        feat = make_dummy_feature("u1", "c1")
        stats = make_dummy_stats()
        sim = make_dummy_sim(avg_sim=-1.0, c_dist=100.0, cons=0.0)
        res = detector.compute_anomaly(feat, stats, sim)
        sim_anom = res["signal_breakdown"]["similarity_anomaly"]
        assert sim_anom > 0.9

    def test_sparse_dense_updates(self, detector: AnomalyDetector):
        feat = make_dummy_feature("u1", "c1")
        stats = make_dummy_stats(spar=1.0) # completely sparse
        sim = make_dummy_sim()
        res = detector.compute_anomaly(feat, stats, sim)
        spar_anom = res["signal_breakdown"]["sparsity_anomaly"]
        assert abs(spar_anom - 1.0) < 0.01

    def test_layer_logic(self, detector: AnomalyDetector):
        feat = make_dummy_feature("u1", "c1")
        stats = make_dummy_stats(layer_l2=100.0, layer_var=100.0)
        sim = make_dummy_sim()
        res = detector.compute_anomaly(feat, stats, sim)
        layer_anom = res["layer_anomaly_scores"]["layer_1"]
        assert layer_anom > 0.9

    def test_composite_score_range(self, detector: AnomalyDetector):
        feat = make_dummy_feature("u1", "c1")
        stats = make_dummy_stats(mag=100.0, rms=100.0, skew=100.0, kurt=100.0, mad=100.0, iqr=100.0, spar=1.0)
        sim = make_dummy_sim(avg_sim=-1.0, c_dist=100.0, cons=0.0)
        res = detector.compute_anomaly(feat, stats, sim)
        assert 0.0 <= res["anomaly_score"] <= 1.0
        assert res["anomaly_score"] > 0.9

class TestAnomalyDetectorEdgeCases:
    """Edge cases logic."""

    def test_missing_data(self, detector: AnomalyDetector):
        with pytest.raises(AnomalyDetectorError):
            detector.compute_anomaly({}, {}, {})

    def test_nan_rejection(self, detector: AnomalyDetector):
        feat = make_dummy_feature("u1", "c1")
        stats = make_dummy_stats(mag=float("nan"))
        sim = make_dummy_sim()
        res = detector.compute_anomaly(feat, stats, sim)
        # NaN is safely clipped to 1.0 by _safe_clip
        assert res["signal_breakdown"]["magnitude_anomaly"] == 1.0

    def test_inf_rejection(self, detector: AnomalyDetector):
        feat = make_dummy_feature("u1", "c1")
        stats = make_dummy_stats()
        sim = make_dummy_sim(avg_sim=float("inf"))
        res = detector.compute_anomaly(feat, stats, sim)
        assert res["signal_breakdown"]["similarity_anomaly"] == 1.0

    def test_constant_zero_huge(self, detector: AnomalyDetector):
        feat = make_dummy_feature("u1", "c1")
        stats = make_dummy_stats(mag=0.0, rms=0.0, skew=0.0, kurt=0.0, mad=0.0, iqr=0.0, spar=0.0, layer_l2=0.0, layer_var=0.0)
        sim = make_dummy_sim(avg_sim=0.0, c_dist=0.0, cons=0.0)
        res = detector.compute_anomaly(feat, stats, sim)
        assert 0.0 <= res["anomaly_score"] <= 1.0

    def test_deterministic(self, detector: AnomalyDetector):
        feat = make_dummy_feature("u1", "c1")
        stats = make_dummy_stats(mag=1.5, skew=0.5, mad=0.2)
        sim = make_dummy_sim(avg_sim=0.8, c_dist=0.1, cons=0.9)
        res1 = detector.compute_anomaly(feat, stats, sim)
        res2 = detector.compute_anomaly(feat, stats, sim)
        assert res1["anomaly_score"] == res2["anomaly_score"]

    def test_no_numpy_leak(self, detector: AnomalyDetector):
        feat = make_dummy_feature("u1", "c1")
        stats = make_dummy_stats()
        sim = make_dummy_sim()
        # Ensure we wrap some values in numpy types to simulate real inputs
        stats["global_statistics"]["update_magnitude"] = np.float64(1.0)
        res = detector.compute_anomaly(feat, stats, sim)
        
        def check_no_numpy(obj):
            if isinstance(obj, dict):
                for v in obj.values():
                    check_no_numpy(v)
            elif isinstance(obj, list):
                for v in obj:
                    check_no_numpy(v)
            else:
                assert not isinstance(obj, np.ndarray)
                assert not isinstance(obj, np.generic)

        check_no_numpy(res)
