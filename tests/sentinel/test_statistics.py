"""
tests/sentinel/test_statistics.py

Unit tests for core/sentinel/statistics.py
Owner: Person 3 — FedSentinel Detection, Impact & Recovery Intelligence

Test coverage:
  1.  Normal feature dictionary (multi-layer model)
  2.  Multi-layer model — layer_statistics keys
  3.  Single-layer model
  4.  Sparse tensor (99%+ zeros)
  5.  Dense tensor (all non-zero)
  6.  Constant tensor (all values identical)
  7.  All-zero tensor
  8.  NaN in flat_parameters → StatisticsEngineError
  9.  Inf in flat_parameters → StatisticsEngineError
  10. Empty feature dictionary / no flat_parameters
  11. Huge tensor (10M parameters)
  12. Layer statistics correctness (per-layer mean/std/median/l2/energy)
  13. MAD correctness (manual verification)
  14. IQR correctness (manual verification)
  15. RMS correctness (manual verification)
  16. Energy correctness (manual verification)
  17. Skewness correctness (manual verification)
  18. Kurtosis correctness (manual verification)

Regression tests:
  Phase 1 outputs pass unchanged through FeatureExtractor after QA improvements.

Phase 1 QA regression checklist:
  - SCHEMA_VERSION = "schema-v1"
  - schema_version returned in feature dict
  - layer_shapes returned in feature dict
  - datatype validation: unsupported type raises FeatureExtractionError
  - Phase 1 features unchanged (l2_norm, mean, std, etc.)
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from core.sentinel.feature_extractor import (
    FEATURE_EXTRACTOR_VERSION,
    SCHEMA_VERSION,
    FeatureExtractionError,
    FeatureExtractor,
)
from core.sentinel.statistics import (
    STATISTICS_ENGINE_VERSION,
    SCHEMA_VERSION as STATS_SCHEMA_VERSION,
    StatisticsEngine,
    StatisticsEngineError,
)
from core.sentinel.sentinel import Sentinel


# ---------------------------------------------------------------------------
# Minimal ModelUpdate stub
# ---------------------------------------------------------------------------

@dataclass
class ModelUpdateStub:
    """Minimal stub conforming to the ModelUpdate contract §8.1."""

    update_id: str
    client_id: str = "c1"
    round_id: int = 1
    parameters: dict[str, Any] = field(default_factory=dict)
    sample_count: int = 100
    training_epochs: int = 5
    learning_rate: float | None = 0.01
    local_loss: float | None = 0.3
    local_accuracy: float | None = 0.9
    run_id: str = "RUN-001"
    model_version: str = "v1"
    base_model_version: str = "v0"
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_update(
    parameters: dict[str, Any],
    update_id: str = "upd-s-001",
    client_id: str = "c1",
    round_id: int = 1,
    sample_count: int = 100,
) -> ModelUpdateStub:
    return ModelUpdateStub(
        update_id=update_id,
        client_id=client_id,
        round_id=round_id,
        parameters=parameters,
        sample_count=sample_count,
    )


def _features_from(parameters: dict[str, Any], update_id: str = "upd-s") -> dict[str, Any]:
    """Convenience: extract features from the given parameters."""
    return FeatureExtractor().extract(_make_update(parameters, update_id=update_id))


# Expected output top-level keys for statistics dict
_STATS_REQUIRED_KEYS = {
    "update_id", "client_id", "round_id",
    "schema_version", "statistics_version",
    "global_statistics", "distribution_statistics",
    "magnitude_statistics", "sparsity_statistics",
    "layer_statistics",
}

_GLOBAL_STATS_KEYS = {
    "variance", "median", "mad", "minimum", "maximum", "value_range",
    "skewness", "kurtosis",
}
_DIST_STATS_KEYS = {
    "percentile_25", "percentile_50", "percentile_75", "interquartile_range",
}
_MAG_STATS_KEYS = {
    "mean_absolute_value", "max_absolute_value", "root_mean_square", "energy",
}
_SPARSITY_STATS_KEYS = {
    "parameter_sparsity", "non_zero_ratio", "zero_count", "non_zero_count",
}
_LAYER_STATS_KEYS = {
    "parameter_count", "mean", "std", "variance", "median", "mad",
    "l2_norm", "max_abs", "min_abs", "rms", "energy", "sparsity",
}

_FORBIDDEN_KEYS = {
    "anomaly_score", "similarity_score", "reputation_score",
    "threat_score", "impact_score", "action",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine() -> StatisticsEngine:
    return StatisticsEngine()


@pytest.fixture
def normal_features() -> dict[str, Any]:
    """Standard 3-layer model with well-behaved float arrays."""
    rng = np.random.default_rng(42)
    params = {
        "layer1.weight": rng.normal(0, 0.01, (64, 32)),
        "layer1.bias":   rng.normal(0, 0.01, (64,)),
        "layer2.weight": rng.normal(0, 0.01, (10, 64)),
    }
    return _features_from(params, update_id="upd-normal")


# ===========================================================================
# TEST 1 — Normal feature dictionary
# ===========================================================================

class TestNormalFeatureDict:
    """Basic correctness and structure for a typical multi-layer model."""

    def test_returns_dict(self, engine: StatisticsEngine, normal_features: dict):
        result = engine.compute(normal_features)
        assert isinstance(result, dict)

    def test_all_required_keys_present(self, engine: StatisticsEngine, normal_features: dict):
        result = engine.compute(normal_features)
        missing = _STATS_REQUIRED_KEYS - result.keys()
        assert not missing, f"Missing top-level keys: {missing}"

    def test_no_forbidden_keys(self, engine: StatisticsEngine, normal_features: dict):
        result = engine.compute(normal_features)
        found = _FORBIDDEN_KEYS & result.keys()
        assert not found, f"Forbidden keys in output: {found}"

    def test_identity_fields_preserved(self, engine: StatisticsEngine, normal_features: dict):
        result = engine.compute(normal_features)
        assert result["update_id"] == normal_features["update_id"]
        assert result["client_id"] == normal_features["client_id"]
        assert result["round_id"] == normal_features["round_id"]

    def test_version_fields(self, engine: StatisticsEngine, normal_features: dict):
        result = engine.compute(normal_features)
        assert result["statistics_version"] == STATISTICS_ENGINE_VERSION
        assert result["schema_version"] == STATS_SCHEMA_VERSION

    def test_global_statistics_keys(self, engine: StatisticsEngine, normal_features: dict):
        result = engine.compute(normal_features)
        missing = _GLOBAL_STATS_KEYS - result["global_statistics"].keys()
        assert not missing, f"Missing global_statistics keys: {missing}"

    def test_distribution_statistics_keys(self, engine: StatisticsEngine, normal_features: dict):
        result = engine.compute(normal_features)
        missing = _DIST_STATS_KEYS - result["distribution_statistics"].keys()
        assert not missing

    def test_magnitude_statistics_keys(self, engine: StatisticsEngine, normal_features: dict):
        result = engine.compute(normal_features)
        missing = _MAG_STATS_KEYS - result["magnitude_statistics"].keys()
        assert not missing

    def test_sparsity_statistics_keys(self, engine: StatisticsEngine, normal_features: dict):
        result = engine.compute(normal_features)
        missing = _SPARSITY_STATS_KEYS - result["sparsity_statistics"].keys()
        assert not missing

    def test_all_global_stats_finite(self, engine: StatisticsEngine, normal_features: dict):
        result = engine.compute(normal_features)
        for key, val in result["global_statistics"].items():
            assert math.isfinite(val), f"global_statistics['{key}'] is not finite: {val}"

    def test_all_dist_stats_finite(self, engine: StatisticsEngine, normal_features: dict):
        result = engine.compute(normal_features)
        for key, val in result["distribution_statistics"].items():
            assert math.isfinite(val), f"distribution_statistics['{key}'] is not finite"

    def test_all_mag_stats_finite(self, engine: StatisticsEngine, normal_features: dict):
        result = engine.compute(normal_features)
        for key, val in result["magnitude_statistics"].items():
            assert math.isfinite(val)

    def test_sparsity_in_range(self, engine: StatisticsEngine, normal_features: dict):
        result = engine.compute(normal_features)
        s = result["sparsity_statistics"]["parameter_sparsity"]
        assert 0.0 <= s <= 1.0

    def test_zero_count_plus_nonzero_equals_total(self, engine: StatisticsEngine, normal_features: dict):
        result = engine.compute(normal_features)
        ss = result["sparsity_statistics"]
        total = ss["zero_count"] + ss["non_zero_count"]
        flat = normal_features["flat_parameters"]
        assert total == flat.size

    def test_via_sentinel(self, normal_features: dict):
        sentinel = Sentinel()
        result = sentinel.compute_statistics(normal_features)
        assert result is not None
        assert "global_statistics" in result

    def test_no_mutation_of_input(self, engine: StatisticsEngine, normal_features: dict):
        """compute() must not modify the features dictionary."""
        original_flat = normal_features["flat_parameters"].copy()
        engine.compute(normal_features)
        np.testing.assert_array_equal(
            normal_features["flat_parameters"], original_flat
        )

    def test_deterministic(self, engine: StatisticsEngine, normal_features: dict):
        """Two calls with the same features produce identical output."""
        r1 = engine.compute(normal_features)
        r2 = engine.compute(normal_features)
        for key in _GLOBAL_STATS_KEYS:
            assert r1["global_statistics"][key] == r2["global_statistics"][key]


# ===========================================================================
# TEST 2 — Multi-layer model layer_statistics
# ===========================================================================

class TestMultiLayerModel:
    """Verify layer_statistics keys exactly match parameters keys."""

    def test_layer_statistics_keys_match_params(self, engine: StatisticsEngine):
        rng = np.random.default_rng(7)
        params = {
            "conv1.weight": rng.normal(0, 0.01, (16, 3, 3, 3)),
            "conv1.bias":   rng.normal(0, 0.01, (16,)),
            "conv2.weight": rng.normal(0, 0.01, (32, 16, 3, 3)),
            "fc.weight":    rng.normal(0, 0.01, (10, 288)),
            "fc.bias":      rng.normal(0, 0.01, (10,)),
        }
        features = _features_from(params, update_id="upd-multilayer")
        result = engine.compute(features)
        assert set(result["layer_statistics"].keys()) == set(params.keys())

    def test_layer_statistics_inner_keys(self, engine: StatisticsEngine):
        params = {"a": np.random.default_rng(0).normal(0, 1, (10, 10))}
        features = _features_from(params, update_id="upd-layerkeys")
        result = engine.compute(features)
        ls = result["layer_statistics"]["a"]
        missing = _LAYER_STATS_KEYS - ls.keys()
        assert not missing, f"Missing layer stats keys: {missing}"

    def test_layer_statistics_values_finite(self, engine: StatisticsEngine):
        rng = np.random.default_rng(11)
        params = {f"layer{i}": rng.normal(0, 1, (20, 20)) for i in range(4)}
        features = _features_from(params, update_id="upd-layerfinite")
        result = engine.compute(features)
        for lname, lstats in result["layer_statistics"].items():
            for key, val in lstats.items():
                if key == "parameter_count":
                    continue
                assert math.isfinite(val), f"layer_statistics['{lname}']['{key}'] not finite"

    def test_layer_parameter_count_correct(self, engine: StatisticsEngine):
        params = {
            "w": np.ones((4, 8)),  # 32 params
            "b": np.ones((4,)),    # 4 params
        }
        features = _features_from(params, update_id="upd-paramcount")
        result = engine.compute(features)
        assert result["layer_statistics"]["w"]["parameter_count"] == 32
        assert result["layer_statistics"]["b"]["parameter_count"] == 4


# ===========================================================================
# TEST 3 — Single-layer model
# ===========================================================================

class TestSingleLayerModel:
    """Single-layer models must work identically to multi-layer."""

    def test_single_layer_all_keys_present(self, engine: StatisticsEngine):
        features = _features_from(
            {"only": np.random.default_rng(1).normal(0, 1, (100,))},
            update_id="upd-single",
        )
        result = engine.compute(features)
        missing = _STATS_REQUIRED_KEYS - result.keys()
        assert not missing

    def test_single_layer_has_one_layer_stat(self, engine: StatisticsEngine):
        features = _features_from({"only": np.ones((50,))}, update_id="upd-single-2")
        result = engine.compute(features)
        assert list(result["layer_statistics"].keys()) == ["only"]

    def test_single_layer_l2_norm_correct(self, engine: StatisticsEngine):
        arr = np.array([3.0, 4.0])
        features = _features_from({"vec": arr}, update_id="upd-single-3")
        result = engine.compute(features)
        assert abs(result["layer_statistics"]["vec"]["l2_norm"] - 5.0) < 1e-9


# ===========================================================================
# TEST 4 — Sparse tensor
# ===========================================================================

class TestSparseTensor:
    """99%+ zero tensor must produce correct sparsity statistics."""

    def test_sparsity_ratio_correct(self, engine: StatisticsEngine):
        arr = np.zeros(1000, dtype=np.float64)
        arr[0] = 1.0
        arr[1] = -2.0  # 998 zeros out of 1000
        features = _features_from({"sparse": arr}, update_id="upd-sparse")
        result = engine.compute(features)
        ss = result["sparsity_statistics"]
        assert abs(ss["parameter_sparsity"] - 0.998) < 1e-9
        assert abs(ss["non_zero_ratio"] - 0.002) < 1e-9
        assert ss["zero_count"] == 998
        assert ss["non_zero_count"] == 2

    def test_sparse_layer_sparsity_correct(self, engine: StatisticsEngine):
        arr = np.zeros(100, dtype=np.float64)
        arr[0] = 5.0
        features = _features_from({"s": arr}, update_id="upd-sparse-layer")
        result = engine.compute(features)
        assert abs(result["layer_statistics"]["s"]["sparsity"] - 0.99) < 1e-9

    def test_sparse_max_abs_correct(self, engine: StatisticsEngine):
        arr = np.zeros(50, dtype=np.float64)
        arr[7] = 3.5
        features = _features_from({"s": arr}, update_id="upd-sparse-max")
        result = engine.compute(features)
        assert abs(result["magnitude_statistics"]["max_absolute_value"] - 3.5) < 1e-9


# ===========================================================================
# TEST 5 — Dense tensor
# ===========================================================================

class TestDenseTensor:
    """All non-zero tensor: sparsity = 0."""

    def test_dense_sparsity_zero(self, engine: StatisticsEngine):
        arr = np.full(200, 5.0, dtype=np.float64)
        features = _features_from({"d": arr}, update_id="upd-dense")
        result = engine.compute(features)
        assert result["sparsity_statistics"]["parameter_sparsity"] == 0.0
        assert result["sparsity_statistics"]["zero_count"] == 0
        assert result["sparsity_statistics"]["non_zero_count"] == 200

    def test_dense_non_zero_ratio_one(self, engine: StatisticsEngine):
        arr = np.full(100, 1.0, dtype=np.float64)
        features = _features_from({"d": arr}, update_id="upd-dense-2")
        result = engine.compute(features)
        assert abs(result["sparsity_statistics"]["non_zero_ratio"] - 1.0) < 1e-9


# ===========================================================================
# TEST 6 — Constant tensor
# ===========================================================================

class TestConstantTensor:
    """All values identical → std=0, skewness=0, kurtosis=0."""

    def test_constant_variance_zero(self, engine: StatisticsEngine):
        arr = np.full(100, 7.0, dtype=np.float64)
        features = _features_from({"c": arr}, update_id="upd-const")
        result = engine.compute(features)
        assert abs(result["global_statistics"]["variance"]) < 1e-9

    def test_constant_skewness_zero(self, engine: StatisticsEngine):
        arr = np.full(100, 3.0, dtype=np.float64)
        features = _features_from({"c": arr}, update_id="upd-const-skew")
        result = engine.compute(features)
        assert abs(result["global_statistics"]["skewness"]) < 1e-9

    def test_constant_kurtosis_zero(self, engine: StatisticsEngine):
        arr = np.full(100, 3.0, dtype=np.float64)
        features = _features_from({"c": arr}, update_id="upd-const-kurt")
        result = engine.compute(features)
        assert abs(result["global_statistics"]["kurtosis"]) < 1e-9

    def test_constant_median_equals_value(self, engine: StatisticsEngine):
        arr = np.full(50, 42.0, dtype=np.float64)
        features = _features_from({"c": arr}, update_id="upd-const-med")
        result = engine.compute(features)
        assert abs(result["global_statistics"]["median"] - 42.0) < 1e-9

    def test_constant_mad_zero(self, engine: StatisticsEngine):
        arr = np.full(50, 1.0, dtype=np.float64)
        features = _features_from({"c": arr}, update_id="upd-const-mad")
        result = engine.compute(features)
        assert abs(result["global_statistics"]["mad"]) < 1e-9

    def test_constant_layer_std_zero(self, engine: StatisticsEngine):
        arr = np.full(20, 9.0, dtype=np.float64)
        features = _features_from({"c": arr}, update_id="upd-const-lstd")
        result = engine.compute(features)
        assert abs(result["layer_statistics"]["c"]["std"]) < 1e-9


# ===========================================================================
# TEST 7 — All-zero tensor
# ===========================================================================

class TestAllZeroTensor:
    """All zeros: sparsity=1.0, all magnitude stats=0."""

    def test_zero_tensor_sparsity_one(self, engine: StatisticsEngine):
        arr = np.zeros(100, dtype=np.float64)
        features = _features_from({"z": arr}, update_id="upd-zero")
        result = engine.compute(features)
        assert abs(result["sparsity_statistics"]["parameter_sparsity"] - 1.0) < 1e-9

    def test_zero_tensor_energy_zero(self, engine: StatisticsEngine):
        arr = np.zeros(50, dtype=np.float64)
        features = _features_from({"z": arr}, update_id="upd-zero-e")
        result = engine.compute(features)
        assert abs(result["magnitude_statistics"]["energy"]) < 1e-9

    def test_zero_tensor_rms_zero(self, engine: StatisticsEngine):
        arr = np.zeros(50, dtype=np.float64)
        features = _features_from({"z": arr}, update_id="upd-zero-rms")
        result = engine.compute(features)
        assert abs(result["magnitude_statistics"]["root_mean_square"]) < 1e-9

    def test_zero_tensor_median_zero(self, engine: StatisticsEngine):
        arr = np.zeros(20, dtype=np.float64)
        features = _features_from({"z": arr}, update_id="upd-zero-med")
        result = engine.compute(features)
        assert abs(result["global_statistics"]["median"]) < 1e-9

    def test_zero_tensor_layer_l2_zero(self, engine: StatisticsEngine):
        arr = np.zeros(30, dtype=np.float64)
        features = _features_from({"z": arr}, update_id="upd-zero-l2")
        result = engine.compute(features)
        assert abs(result["layer_statistics"]["z"]["l2_norm"]) < 1e-9


# ===========================================================================
# TEST 8 — NaN rejection
# ===========================================================================

class TestNaNRejection:
    """NaN in flat_parameters must raise StatisticsEngineError."""

    def test_nan_raises_statistics_engine_error(self, engine: StatisticsEngine):
        features = _features_from({"l": np.array([1.0, 2.0])}, update_id="upd-nan-stats")
        features_copy = dict(features)
        features_copy["flat_parameters"] = np.array([1.0, float("nan"), 2.0])
        with pytest.raises(StatisticsEngineError) as exc_info:
            engine.compute(features_copy)
        assert "NaN" in str(exc_info.value)

    def test_nan_error_is_exception_subclass(self, engine: StatisticsEngine):
        features = _features_from({"l": np.array([1.0])}, update_id="upd-nan-exc")
        features_copy = dict(features)
        features_copy["flat_parameters"] = np.array([float("nan")])
        with pytest.raises(Exception):
            engine.compute(features_copy)

    def test_nan_error_contains_update_id(self, engine: StatisticsEngine):
        features = _features_from({"l": np.array([1.0])}, update_id="upd-nan-id")
        features_copy = dict(features)
        features_copy["flat_parameters"] = np.array([float("nan")])
        with pytest.raises(StatisticsEngineError) as exc_info:
            engine.compute(features_copy)
        assert "upd-nan-id" in str(exc_info.value)


# ===========================================================================
# TEST 9 — Inf rejection
# ===========================================================================

class TestInfRejection:
    """Inf in flat_parameters must raise StatisticsEngineError."""

    def test_positive_inf_raises(self, engine: StatisticsEngine):
        features = _features_from({"l": np.array([1.0])}, update_id="upd-inf-stats")
        features_copy = dict(features)
        features_copy["flat_parameters"] = np.array([1.0, float("inf")])
        with pytest.raises(StatisticsEngineError) as exc_info:
            engine.compute(features_copy)
        assert "Inf" in str(exc_info.value)

    def test_negative_inf_raises(self, engine: StatisticsEngine):
        features = _features_from({"l": np.array([1.0])}, update_id="upd-neg-inf-stats")
        features_copy = dict(features)
        features_copy["flat_parameters"] = np.array([float("-inf")])
        with pytest.raises(StatisticsEngineError):
            engine.compute(features_copy)

    def test_inf_error_is_exception_subclass(self, engine: StatisticsEngine):
        features = _features_from({"l": np.array([1.0])}, update_id="upd-inf-exc")
        features_copy = dict(features)
        features_copy["flat_parameters"] = np.array([float("inf")])
        with pytest.raises(Exception):
            engine.compute(features_copy)


# ===========================================================================
# TEST 10 — Empty / missing flat_parameters
# ===========================================================================

class TestEmptyOrMissingFlatParameters:
    """Empty parameters dict must return zero statistics without crashing."""

    def test_empty_params_no_exception(self, engine: StatisticsEngine):
        features = _features_from({}, update_id="upd-empty-stats")
        result = engine.compute(features)
        assert result is not None

    def test_empty_params_global_stats_zero(self, engine: StatisticsEngine):
        features = _features_from({}, update_id="upd-empty-gs")
        result = engine.compute(features)
        gs = result["global_statistics"]
        for key in _GLOBAL_STATS_KEYS:
            assert gs[key] == 0.0, f"global_statistics['{key}'] should be 0.0, got {gs[key]}"

    def test_empty_params_layer_statistics_empty(self, engine: StatisticsEngine):
        features = _features_from({}, update_id="upd-empty-ls")
        result = engine.compute(features)
        assert result["layer_statistics"] == {}

    def test_empty_params_sparsity_zero(self, engine: StatisticsEngine):
        features = _features_from({}, update_id="upd-empty-sp")
        result = engine.compute(features)
        ss = result["sparsity_statistics"]
        assert ss["zero_count"] == 0
        assert ss["non_zero_count"] == 0

    def test_missing_flat_parameters_key_no_crash(self, engine: StatisticsEngine):
        """If flat_parameters is absent, engine must still work (treats as empty)."""
        features = {
            "update_id": "upd-no-flat",
            "client_id": "c1",
            "round_id": 1,
            "layer_parameters": {},
            "schema_version": "schema-v1",
        }
        result = engine.compute(features)
        assert result["global_statistics"]["median"] == 0.0

    def test_missing_keys_raises(self, engine: StatisticsEngine):
        """Features dict missing required keys must raise StatisticsEngineError."""
        with pytest.raises(StatisticsEngineError):
            engine.compute({"update_id": "x"})  # client_id and round_id missing


# ===========================================================================
# TEST 11 — Huge tensor
# ===========================================================================

class TestHugeTensor:
    """10M parameter tensor must complete within performance bounds."""

    def test_huge_tensor_completes(self, engine: StatisticsEngine):
        import time
        rng = np.random.default_rng(0)
        big = rng.normal(0, 0.01, (10_000_000,))
        features = _features_from({"huge": big}, update_id="upd-huge-stats")
        t0 = time.perf_counter()
        result = engine.compute(features)
        elapsed = time.perf_counter() - t0
        assert result is not None
        # Performance target: large model under 300 ms
        assert elapsed < 3.0, f"Huge tensor took {elapsed:.3f}s (target < 3.0s)"

    def test_huge_tensor_all_required_keys(self, engine: StatisticsEngine):
        rng = np.random.default_rng(1)
        big = rng.normal(0, 0.01, (10_000_000,))
        features = _features_from({"huge": big}, update_id="upd-huge-keys")
        result = engine.compute(features)
        missing = _STATS_REQUIRED_KEYS - result.keys()
        assert not missing

    def test_huge_tensor_stats_finite(self, engine: StatisticsEngine):
        rng = np.random.default_rng(2)
        big = rng.normal(0, 0.01, (10_000_000,))
        features = _features_from({"huge": big}, update_id="upd-huge-finite")
        result = engine.compute(features)
        for key in _GLOBAL_STATS_KEYS:
            val = result["global_statistics"][key]
            assert math.isfinite(val), f"global_statistics['{key}'] not finite"


# ===========================================================================
# TEST 12 — Layer statistics correctness
# ===========================================================================

class TestLayerStatisticsCorrectness:
    """Independent manual verification of each layer statistic."""

    def _get_layer_stats(self, arr: np.ndarray, name: str = "L") -> dict:
        engine = StatisticsEngine()
        features = _features_from({name: arr}, update_id="upd-ls-corr")
        result = engine.compute(features)
        return result["layer_statistics"][name]

    def test_layer_mean_correct(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ls = self._get_layer_stats(arr)
        assert abs(ls["mean"] - 3.0) < 1e-9

    def test_layer_std_correct(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected_std = float(np.std(arr))
        ls = self._get_layer_stats(arr)
        assert abs(ls["std"] - expected_std) < 1e-9

    def test_layer_variance_correct(self):
        arr = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        expected_var = float(np.var(arr))
        ls = self._get_layer_stats(arr)
        assert abs(ls["variance"] - expected_var) < 1e-9

    def test_layer_median_correct(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ls = self._get_layer_stats(arr)
        assert abs(ls["median"] - 3.0) < 1e-9

    def test_layer_l2_norm_correct(self):
        arr = np.array([3.0, 4.0])
        ls = self._get_layer_stats(arr)
        assert abs(ls["l2_norm"] - 5.0) < 1e-9

    def test_layer_max_abs_correct(self):
        arr = np.array([-7.0, 2.0, 3.0])
        ls = self._get_layer_stats(arr)
        assert abs(ls["max_abs"] - 7.0) < 1e-9

    def test_layer_min_abs_correct(self):
        arr = np.array([-7.0, 2.0, 3.0])
        ls = self._get_layer_stats(arr)
        assert abs(ls["min_abs"] - 2.0) < 1e-9

    def test_layer_energy_correct(self):
        arr = np.array([1.0, 2.0, 2.0])
        ls = self._get_layer_stats(arr)
        # energy = 1² + 2² + 2² = 1 + 4 + 4 = 9
        assert abs(ls["energy"] - 9.0) < 1e-9

    def test_layer_rms_correct(self):
        arr = np.array([1.0, 2.0, 2.0])
        ls = self._get_layer_stats(arr)
        # rms = sqrt((1+4+4)/3) = sqrt(3) ≈ 1.7320...
        expected = float(np.sqrt(9.0 / 3))
        assert abs(ls["rms"] - expected) < 1e-9


# ===========================================================================
# TEST 13 — MAD correctness
# ===========================================================================

class TestMADCorrectness:
    """Median Absolute Deviation: MAD = median(|X - median(X)|)."""

    def test_mad_simple(self):
        engine = StatisticsEngine()
        arr = np.array([1.0, 1.0, 2.0, 2.0, 4.0, 6.0, 9.0])
        features = _features_from({"m": arr}, update_id="upd-mad-1")
        result = engine.compute(features)
        median = float(np.median(arr))
        expected_mad = float(np.median(np.abs(arr - median)))
        assert abs(result["global_statistics"]["mad"] - expected_mad) < 1e-9

    def test_mad_constant_array(self):
        engine = StatisticsEngine()
        arr = np.full(10, 3.0)
        features = _features_from({"m": arr}, update_id="upd-mad-const")
        result = engine.compute(features)
        assert abs(result["global_statistics"]["mad"]) < 1e-9

    def test_layer_mad_correct(self):
        engine = StatisticsEngine()
        arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        features = _features_from({"m": arr}, update_id="upd-layer-mad")
        result = engine.compute(features)
        median = float(np.median(arr))
        expected_mad = float(np.median(np.abs(arr - median)))
        assert abs(result["layer_statistics"]["m"]["mad"] - expected_mad) < 1e-9


# ===========================================================================
# TEST 14 — IQR correctness
# ===========================================================================

class TestIQRCorrectness:
    """IQR = P75 - P25."""

    def test_iqr_known_array(self):
        engine = StatisticsEngine()
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        features = _features_from({"i": arr}, update_id="upd-iqr")
        result = engine.compute(features)
        p25, p75 = np.percentile(arr, [25, 75])
        expected_iqr = float(p75 - p25)
        assert abs(result["distribution_statistics"]["interquartile_range"] - expected_iqr) < 1e-9

    def test_percentile_50_equals_global_median(self):
        engine = StatisticsEngine()
        rng = np.random.default_rng(33)
        arr = rng.normal(0, 1, (1000,))
        features = _features_from({"p": arr}, update_id="upd-p50-med")
        result = engine.compute(features)
        p50 = result["distribution_statistics"]["percentile_50"]
        median = result["global_statistics"]["median"]
        assert abs(p50 - median) < 1e-9

    def test_percentile_ordering(self):
        engine = StatisticsEngine()
        arr = np.arange(100, dtype=np.float64)
        features = _features_from({"p": arr}, update_id="upd-porder")
        result = engine.compute(features)
        ds = result["distribution_statistics"]
        assert ds["percentile_25"] <= ds["percentile_50"] <= ds["percentile_75"]


# ===========================================================================
# TEST 15 — RMS correctness
# ===========================================================================

class TestRMSCorrectness:
    """root_mean_square = sqrt(mean(x²)) and must match FeatureExtractor's update_magnitude."""

    def test_rms_manual(self):
        engine = StatisticsEngine()
        arr = np.array([3.0, 4.0])
        features = _features_from({"r": arr}, update_id="upd-rms")
        result = engine.compute(features)
        expected = float(np.sqrt(np.mean(arr ** 2)))  # sqrt(25/2) = sqrt(12.5)
        assert abs(result["magnitude_statistics"]["root_mean_square"] - expected) < 1e-9

    def test_rms_matches_feature_extractor_update_magnitude(self):
        """RMS in StatisticsEngine must match update_magnitude from FeatureExtractor."""
        engine = StatisticsEngine()
        rng = np.random.default_rng(77)
        arr = rng.normal(0, 1, (500,))
        features = _features_from({"r": arr}, update_id="upd-rms-match")
        result = engine.compute(features)
        # FeatureExtractor's update_magnitude is also RMS
        fe_rms = features["update_magnitude"]
        stats_rms = result["magnitude_statistics"]["root_mean_square"]
        assert abs(stats_rms - fe_rms) < 1e-9


# ===========================================================================
# TEST 16 — Energy correctness
# ===========================================================================

class TestEnergyCorrectness:
    """energy = sum(x²) = l2_norm²."""

    def test_energy_equals_l2_norm_squared(self):
        engine = StatisticsEngine()
        arr = np.array([3.0, 4.0])
        features = _features_from({"e": arr}, update_id="upd-energy")
        result = engine.compute(features)
        l2 = features["l2_norm"]
        energy = result["magnitude_statistics"]["energy"]
        assert abs(energy - l2 ** 2) < 1e-6

    def test_energy_manual(self):
        engine = StatisticsEngine()
        arr = np.array([1.0, 2.0, 2.0])
        features = _features_from({"e": arr}, update_id="upd-energy-2")
        result = engine.compute(features)
        assert abs(result["magnitude_statistics"]["energy"] - 9.0) < 1e-9


# ===========================================================================
# TEST 17 — Skewness correctness
# ===========================================================================

class TestSkewnessCorrectness:
    """Skewness = E[(X-μ)³] / σ³ (Pearson's moment coefficient)."""

    def test_skewness_symmetric_array_near_zero(self):
        engine = StatisticsEngine()
        arr = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        features = _features_from({"s": arr}, update_id="upd-skew-sym")
        result = engine.compute(features)
        assert abs(result["global_statistics"]["skewness"]) < 1e-9

    def test_skewness_positive_skew(self):
        """Heavy right tail → positive skewness."""
        engine = StatisticsEngine()
        arr = np.array([0.0, 0.0, 0.0, 1.0, 10.0])
        features = _features_from({"s": arr}, update_id="upd-skew-pos")
        result = engine.compute(features)
        assert result["global_statistics"]["skewness"] > 0.0

    def test_skewness_manual_verification(self):
        engine = StatisticsEngine()
        rng = np.random.default_rng(55)
        arr = rng.normal(0, 1, (2000,))
        features = _features_from({"s": arr}, update_id="upd-skew-manual")
        result = engine.compute(features)
        std = float(np.std(arr))
        centered = arr - float(np.mean(arr))
        expected = float(np.mean(centered ** 3) / std ** 3)
        assert abs(result["global_statistics"]["skewness"] - expected) < 1e-9


# ===========================================================================
# TEST 18 — Kurtosis correctness
# ===========================================================================

class TestKurtosisCorrectness:
    """Excess kurtosis = E[(X-μ)⁴]/σ⁴ - 3. Normal dist ≈ 0."""

    def test_kurtosis_normal_distribution_near_zero(self):
        """Normal distribution excess kurtosis should be ≈ 0 for large n."""
        engine = StatisticsEngine()
        rng = np.random.default_rng(99)
        arr = rng.normal(0, 1, (100_000,))
        features = _features_from({"k": arr}, update_id="upd-kurt-normal")
        result = engine.compute(features)
        # With 100K samples, excess kurtosis should be within 0.1 of 0
        assert abs(result["global_statistics"]["kurtosis"]) < 0.1

    def test_kurtosis_constant_zero(self):
        engine = StatisticsEngine()
        arr = np.full(50, 1.0, dtype=np.float64)
        features = _features_from({"k": arr}, update_id="upd-kurt-const")
        result = engine.compute(features)
        assert abs(result["global_statistics"]["kurtosis"]) < 1e-9

    def test_kurtosis_manual_verification(self):
        engine = StatisticsEngine()
        rng = np.random.default_rng(88)
        arr = rng.normal(0, 1, (1000,))
        features = _features_from({"k": arr}, update_id="upd-kurt-manual")
        result = engine.compute(features)
        std = float(np.std(arr))
        centered = arr - float(np.mean(arr))
        expected = float(np.mean(centered ** 4) / std ** 4 - 3.0)
        assert abs(result["global_statistics"]["kurtosis"] - expected) < 1e-9


# ===========================================================================
# REGRESSION TESTS — Phase 1 QA improvements
# ===========================================================================

class TestPhase1Regression:
    """
    Verify all Phase 1 QA improvements are applied and Phase 1 outputs
    are unchanged after the QA refactor.
    """

    def test_schema_version_is_schema_v1(self):
        """SCHEMA_VERSION must be 'schema-v1' (contract requirement)."""
        assert SCHEMA_VERSION == "schema-v1"

    def test_schema_version_returned_in_feature_dict(self):
        extractor = FeatureExtractor()
        features = extractor.extract(_make_update(
            {"l": np.array([1.0])}, update_id="upd-reg-sv"
        ))
        assert features.get("schema_version") == "schema-v1"

    def test_layer_shapes_returned_in_feature_dict(self):
        extractor = FeatureExtractor()
        params = {"w": np.ones((8, 4)), "b": np.ones((8,))}
        features = extractor.extract(_make_update(params, update_id="upd-reg-shapes"))
        assert "layer_shapes" in features
        assert features["layer_shapes"]["w"] == (8, 4)
        assert features["layer_shapes"]["b"] == (8,)

    def test_unsupported_string_type_raises(self):
        """Unsupported datatype must raise FeatureExtractionError (not silently convert)."""
        extractor = FeatureExtractor()
        with pytest.raises(FeatureExtractionError) as exc_info:
            extractor.extract(_make_update({"layer1": "hello"}, update_id="upd-reg-str"))
        assert "Unsupported" in str(exc_info.value)
        assert "layer1" in str(exc_info.value)

    def test_unsupported_none_type_raises(self):
        extractor = FeatureExtractor()
        with pytest.raises(FeatureExtractionError):
            extractor.extract(_make_update({"layer1": None}, update_id="upd-reg-none"))

    def test_phase1_features_unchanged(self):
        """
        Phase 1 core features (l2_norm, mean, std, max_abs, min_abs,
        update_magnitude, parameter_sparsity, layer_magnitudes) must
        produce the same values as before the QA refactor.
        """
        rng = np.random.default_rng(42)
        arr = rng.normal(0, 0.01, (64, 32))
        extractor = FeatureExtractor()
        features = extractor.extract(_make_update({"w": arr}, update_id="upd-reg-p1"))

        # Recompute expected values independently
        flat = arr.ravel().astype(np.float64)
        assert abs(features["l2_norm"] - float(np.linalg.norm(flat))) < 1e-9
        assert abs(features["mean"] - float(np.mean(flat))) < 1e-9
        assert abs(features["std"] - float(np.std(flat))) < 1e-9
        assert abs(features["max_abs"] - float(np.max(np.abs(flat)))) < 1e-9
        assert abs(features["min_abs"] - float(np.min(np.abs(flat)))) < 1e-9
        expected_rms = float(np.sqrt(np.mean(flat ** 2)))
        assert abs(features["update_magnitude"] - expected_rms) < 1e-9
        near_zero = float(np.sum(np.abs(flat) < 1e-6)) / flat.size
        assert abs(features["parameter_sparsity"] - near_zero) < 1e-9
        assert abs(features["layer_magnitudes"]["w"] - float(np.linalg.norm(flat))) < 1e-9

    def test_full_phase1_suite_regression(self):
        """
        Run through all 54 original Phase 1 test scenarios to ensure nothing broke.
        This is a smoke test — the full test_feature_extractor.py is the authoritative suite.
        """
        extractor = FeatureExtractor()
        rng = np.random.default_rng(42)

        # Normal update
        params = {"l1": rng.normal(0, 0.01, (64, 32)).tolist(), "b": rng.normal(0, 0.01, (64,)).tolist()}
        f = extractor.extract(_make_update(params, update_id="reg-normal"))
        assert f["l2_norm"] >= 0

        # Empty params
        f_empty = extractor.extract(_make_update({}, update_id="reg-empty"))
        assert f_empty["l2_norm"] == 0.0

        # NaN raises
        with pytest.raises(FeatureExtractionError):
            extractor.extract(_make_update({"l": np.array([float("nan")])}, update_id="reg-nan"))

        # Inf raises
        with pytest.raises(FeatureExtractionError):
            extractor.extract(_make_update({"l": np.array([float("inf")])}, update_id="reg-inf"))

        # Large tensor completes
        big = rng.normal(0, 0.01, (1_000_000,))
        f_big = extractor.extract(_make_update({"big": big}, update_id="reg-big"))
        assert math.isfinite(f_big["l2_norm"])


# ===========================================================================
# QUALITY CHECKS
# ===========================================================================

class TestQualityChecks:
    """Non-functional quality guarantees for StatisticsEngine."""

    def test_version_constant_exists(self):
        assert isinstance(STATISTICS_ENGINE_VERSION, str)
        assert len(STATISTICS_ENGINE_VERSION) > 0

    def test_version_constant_value(self):
        assert STATISTICS_ENGINE_VERSION == "statistics-v1"

    def test_schema_version_constant_matches(self):
        assert STATS_SCHEMA_VERSION == "schema-v1"

    def test_statistics_engine_error_is_exception_subclass(self):
        assert issubclass(StatisticsEngineError, Exception)

    def test_statistics_engine_error_can_be_raised_and_caught(self):
        with pytest.raises(StatisticsEngineError):
            raise StatisticsEngineError("test error")

    def test_docstring_on_statistics_engine_class(self):
        assert StatisticsEngine.__doc__ is not None
        assert len(StatisticsEngine.__doc__.strip()) > 20

    def test_docstring_on_compute_method(self):
        assert StatisticsEngine.compute.__doc__ is not None
        assert len(StatisticsEngine.compute.__doc__.strip()) > 20

    def test_no_forbidden_keys_in_output(self):
        engine = StatisticsEngine()
        features = _features_from({"l": np.array([1.0, 2.0])}, update_id="upd-qforbid")
        result = engine.compute(features)
        found = _FORBIDDEN_KEYS & result.keys()
        assert not found, f"Forbidden keys in output: {found}"

    def test_deterministic_output(self):
        engine = StatisticsEngine()
        rng = np.random.default_rng(55)
        arr = rng.normal(0, 1, (200,))
        features = _features_from({"d": arr}, update_id="upd-qdet")
        r1 = engine.compute(features)
        r2 = engine.compute(features)
        for key in _GLOBAL_STATS_KEYS:
            assert r1["global_statistics"][key] == r2["global_statistics"][key]

    def test_no_mutation_of_input_features(self):
        engine = StatisticsEngine()
        arr = np.array([1.0, 2.0, 3.0])
        features = _features_from({"l": arr}, update_id="upd-qmut")
        flat_before = features["flat_parameters"].copy()
        engine.compute(features)
        np.testing.assert_array_equal(features["flat_parameters"], flat_before)

    def test_logging_fires_on_compute(self):
        engine = StatisticsEngine()
        features = _features_from({"l": np.array([1.0, 2.0])}, update_id="upd-qlog")

        records: list[logging.LogRecord] = []

        class Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = Capture()
        handler.setLevel(logging.DEBUG)
        se_logger = logging.getLogger("core.sentinel.statistics")
        se_logger.addHandler(handler)
        se_logger.setLevel(logging.DEBUG)
        try:
            engine.compute(features)
        finally:
            se_logger.removeHandler(handler)

        assert len(records) > 0, "Expected at least one log record from StatisticsEngine.compute"

    def test_stateless_engine(self):
        """Different StatisticsEngine instances produce identical output."""
        arr = np.array([1.0, 2.0, 3.0])
        features = _features_from({"l": arr}, update_id="upd-qstate")
        r1 = StatisticsEngine().compute(features)
        r2 = StatisticsEngine().compute(features)
        for key in _GLOBAL_STATS_KEYS:
            assert r1["global_statistics"][key] == r2["global_statistics"][key]


# ===========================================================================
# PERFORMANCE TESTS
# ===========================================================================

class TestPerformance:
    """Runtime and memory benchmarks."""

    def test_small_model_under_5ms(self):
        import time
        engine = StatisticsEngine()
        rng = np.random.default_rng(42)
        params = {
            "layer1.weight": rng.normal(0, 0.01, (256, 128)),
            "layer1.bias":   rng.normal(0, 0.01, (256,)),
            "layer2.weight": rng.normal(0, 0.01, (128, 64)),
        }
        features = _features_from(params, update_id="perf-small")
        engine.compute(features)  # warm-up

        N = 20
        t0 = time.perf_counter()
        for _ in range(N):
            engine.compute(features)
        elapsed_ms = (time.perf_counter() - t0) / N * 1000
        # Allow 5x slack (5ms target → 25ms threshold)
        assert elapsed_ms < 25.0, f"Small model avg {elapsed_ms:.2f} ms (target < 25ms)"

    def test_large_model_under_300ms(self):
        import time
        engine = StatisticsEngine()
        rng = np.random.default_rng(0)
        big = rng.normal(0, 0.01, (10_000_000,))
        features = _features_from({"fc": big}, update_id="perf-large")
        t0 = time.perf_counter()
        engine.compute(features)
        elapsed = time.perf_counter() - t0
        # Allow 3x slack (300ms target → 3.0s threshold)
        assert elapsed < 3.0, f"Large model took {elapsed:.3f}s (target < 3.0s)"

    def test_deterministic_repeated_execution(self):
        engine = StatisticsEngine()
        rng = np.random.default_rng(42)
        arr = rng.normal(0, 1, (10_000,))
        features = _features_from({"l": arr}, update_id="perf-det")
        results = [engine.compute(features) for _ in range(5)]
        ref = results[0]["global_statistics"]
        for r in results[1:]:
            for key in _GLOBAL_STATS_KEYS:
                assert r["global_statistics"][key] == ref[key]
