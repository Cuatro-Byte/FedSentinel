"""
tests/sentinel/test_feature_extractor.py

Unit tests for core/sentinel/feature_extractor.py
Owner: Person 3 — FedSentinel Detection, Impact & Recovery Intelligence

Test coverage:
  1. Normal ModelUpdate (multi-layer)
  2. Multi-layer update — verify layer_magnitudes keys
  3. Single-layer update
  4. Empty parameters dict
  5. NaN parameters → FeatureExtractionError
  6. Inf parameters → FeatureExtractionError
  7. Zero sample_count — valid result
  8. Very large tensor (10M parameters)

Quality checks:
  - Deterministic output (two identical runs produce identical output)
  - No mutation of input update
  - FEATURE_EXTRACTOR_VERSION constant exists and is a non-empty string
  - FeatureExtractionError is a subclass of Exception
  - Logging fires (captured via logging.handlers.MemoryHandler)
  - Docstrings present on FeatureExtractor class and public methods
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
from core.sentinel.sentinel import Sentinel


# ---------------------------------------------------------------------------
# Minimal ModelUpdate stub — used ONLY for testing.
# Production code will import from core.models.model_update (Person 1).
# ---------------------------------------------------------------------------

@dataclass
class ModelUpdateStub:
    """Minimal stub conforming to the ModelUpdate contract §8.1."""

    update_id: str
    client_id: str
    round_id: int
    parameters: dict[str, Any]
    sample_count: int
    training_epochs: int
    learning_rate: float | None
    local_loss: float | None
    local_accuracy: float | None
    # Additional contract fields (not used by feature extractor but present)
    run_id: str = "RUN-001"
    model_version: str = "v1"
    base_model_version: str = "v0"
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_update(
    parameters: dict[str, Any],
    update_id: str = "upd-test-001",
    client_id: str = "client-01",
    round_id: int = 1,
    sample_count: int = 100,
    training_epochs: int = 5,
    learning_rate: float | None = 0.01,
    local_loss: float | None = 0.42,
    local_accuracy: float | None = 0.91,
) -> ModelUpdateStub:
    return ModelUpdateStub(
        update_id=update_id,
        client_id=client_id,
        round_id=round_id,
        parameters=parameters,
        sample_count=sample_count,
        training_epochs=training_epochs,
        learning_rate=learning_rate,
        local_loss=local_loss,
        local_accuracy=local_accuracy,
    )


# ---------------------------------------------------------------------------
# Expected output key sets
# ---------------------------------------------------------------------------

_METADATA_KEYS = {
    "update_id", "client_id", "round_id", "sample_count",
    "training_epochs", "learning_rate", "local_loss", "local_accuracy",
}
_FEATURE_KEYS = {
    "l2_norm", "mean", "std", "max_abs", "min_abs",
    "update_magnitude", "parameter_sparsity", "layer_magnitudes",
}
# Phase 1 QA additions
_PHASE1_QA_KEYS = {"layer_shapes", "flat_parameters", "layer_parameters"}
_ALL_REQUIRED_KEYS = _METADATA_KEYS | _FEATURE_KEYS | {"extractor_version"} | _PHASE1_QA_KEYS

_FORBIDDEN_KEYS = {
    "anomaly_score", "threat_score", "similarity_score",
    "reputation", "impact_score", "response_action",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def extractor() -> FeatureExtractor:
    return FeatureExtractor()


@pytest.fixture
def normal_update() -> ModelUpdateStub:
    """Standard 3-layer update with well-behaved float arrays."""
    rng = np.random.default_rng(42)
    return _make_update(
        parameters={
            "layer1.weight": rng.normal(0, 0.01, (64, 32)).tolist(),
            "layer1.bias": rng.normal(0, 0.01, (64,)).tolist(),
            "layer2.weight": rng.normal(0, 0.01, (10, 64)).tolist(),
        }
    )


# ===========================================================================
# TEST 1 — Normal ModelUpdate
# ===========================================================================

class TestNormalUpdate:
    """Verify correct feature extraction for a typical multi-layer update."""

    def test_returns_dict(self, extractor: FeatureExtractor, normal_update: ModelUpdateStub):
        result = extractor.extract(normal_update)
        assert isinstance(result, dict)

    def test_all_required_keys_present(self, extractor: FeatureExtractor, normal_update: ModelUpdateStub):
        result = extractor.extract(normal_update)
        missing = _ALL_REQUIRED_KEYS - result.keys()
        assert not missing, f"Missing keys in output: {missing}"

    def test_no_forbidden_keys_present(self, extractor: FeatureExtractor, normal_update: ModelUpdateStub):
        result = extractor.extract(normal_update)
        found_forbidden = _FORBIDDEN_KEYS & result.keys()
        assert not found_forbidden, f"Forbidden keys found in output: {found_forbidden}"

    def test_metadata_values_preserved(self, extractor: FeatureExtractor, normal_update: ModelUpdateStub):
        result = extractor.extract(normal_update)
        assert result["update_id"] == normal_update.update_id
        assert result["client_id"] == normal_update.client_id
        assert result["round_id"] == normal_update.round_id
        assert result["sample_count"] == normal_update.sample_count
        assert result["training_epochs"] == normal_update.training_epochs
        assert result["learning_rate"] == normal_update.learning_rate
        assert result["local_loss"] == normal_update.local_loss
        assert result["local_accuracy"] == normal_update.local_accuracy

    def test_numeric_features_are_finite(self, extractor: FeatureExtractor, normal_update: ModelUpdateStub):
        result = extractor.extract(normal_update)
        for key in ["l2_norm", "mean", "std", "max_abs", "min_abs",
                    "update_magnitude", "parameter_sparsity"]:
            assert math.isfinite(result[key]), f"Feature '{key}' is not finite: {result[key]}"

    def test_l2_norm_positive(self, extractor: FeatureExtractor, normal_update: ModelUpdateStub):
        result = extractor.extract(normal_update)
        assert result["l2_norm"] >= 0.0

    def test_sparsity_in_range(self, extractor: FeatureExtractor, normal_update: ModelUpdateStub):
        result = extractor.extract(normal_update)
        assert 0.0 <= result["parameter_sparsity"] <= 1.0

    def test_extractor_version_matches_constant(self, extractor: FeatureExtractor, normal_update: ModelUpdateStub):
        result = extractor.extract(normal_update)
        assert result["extractor_version"] == FEATURE_EXTRACTOR_VERSION

    def test_layer_magnitudes_is_dict(self, extractor: FeatureExtractor, normal_update: ModelUpdateStub):
        result = extractor.extract(normal_update)
        assert isinstance(result["layer_magnitudes"], dict)

    def test_layer_magnitudes_values_are_finite(self, extractor: FeatureExtractor, normal_update: ModelUpdateStub):
        result = extractor.extract(normal_update)
        for layer, mag in result["layer_magnitudes"].items():
            assert math.isfinite(mag), f"layer_magnitudes['{layer}'] is not finite: {mag}"

    def test_update_magnitude_equals_rms(self, extractor: FeatureExtractor, normal_update: ModelUpdateStub):
        """update_magnitude should equal RMS of all parameters."""
        result = extractor.extract(normal_update)
        # Recompute manually
        all_vals = []
        for tensor in normal_update.parameters.values():
            all_vals.extend(np.array(tensor).ravel().tolist())
        arr = np.array(all_vals, dtype=np.float64)
        expected_rms = float(np.sqrt(np.mean(arr ** 2)))
        assert abs(result["update_magnitude"] - expected_rms) < 1e-9

    def test_via_sentinel(self, normal_update: ModelUpdateStub):
        """Sentinel.extract_features should produce identical scalar output to FeatureExtractor."""
        extractor = FeatureExtractor()
        sentinel = Sentinel()
        direct = extractor.extract(normal_update)
        via_sentinel = sentinel.extract_features(normal_update)

        # Compare scalar / non-array fields
        scalar_keys = [
            "update_id", "client_id", "round_id", "sample_count",
            "training_epochs", "learning_rate", "local_loss", "local_accuracy",
            "l2_norm", "mean", "std", "max_abs", "min_abs",
            "update_magnitude", "parameter_sparsity",
            "layer_magnitudes", "layer_shapes",
            "extractor_version", "schema_version",
        ]
        for key in scalar_keys:
            assert direct[key] == via_sentinel[key], f"Mismatch on key '{key}'"

        # Compare numpy array fields element-wise
        np.testing.assert_array_equal(direct["flat_parameters"], via_sentinel["flat_parameters"])
        for layer_name in direct["layer_parameters"]:
            np.testing.assert_array_equal(
                direct["layer_parameters"][layer_name],
                via_sentinel["layer_parameters"][layer_name],
            )


# ===========================================================================
# TEST 2 — Multi-layer update
# ===========================================================================

class TestMultiLayerUpdate:
    """Verify layer_magnitudes keys exactly match parameters keys."""

    def test_layer_magnitudes_keys_match_parameters(self, extractor: FeatureExtractor):
        rng = np.random.default_rng(7)
        params = {
            "conv1.weight": rng.normal(0, 0.01, (16, 3, 3, 3)),
            "conv1.bias": rng.normal(0, 0.01, (16,)),
            "conv2.weight": rng.normal(0, 0.01, (32, 16, 3, 3)),
            "conv2.bias": rng.normal(0, 0.01, (32,)),
            "fc.weight": rng.normal(0, 0.01, (10, 288)),
            "fc.bias": rng.normal(0, 0.01, (10,)),
        }
        update = _make_update(params, update_id="upd-multilayer")
        result = extractor.extract(update)
        assert set(result["layer_magnitudes"].keys()) == set(params.keys())

    def test_layer_magnitudes_values_nonneg(self, extractor: FeatureExtractor):
        rng = np.random.default_rng(99)
        params = {f"layer{i}": rng.normal(0, 1, (50, 50)) for i in range(5)}
        update = _make_update(params, update_id="upd-multilayer-2")
        result = extractor.extract(update)
        for layer, mag in result["layer_magnitudes"].items():
            assert mag >= 0.0, f"Negative magnitude for layer '{layer}'"

    def test_layer_magnitudes_each_correct(self, extractor: FeatureExtractor):
        """Each layer magnitude should match np.linalg.norm of that layer."""
        rng = np.random.default_rng(13)
        params = {
            "A": rng.normal(0, 1, (10, 10)),
            "B": rng.normal(0, 1, (5,)),
        }
        update = _make_update(params, update_id="upd-layermag-verify")
        result = extractor.extract(update)
        for name, tensor in params.items():
            expected = float(np.linalg.norm(np.array(tensor).ravel()))
            actual = result["layer_magnitudes"][name]
            assert abs(actual - expected) < 1e-9, (
                f"layer_magnitudes['{name}']: expected {expected}, got {actual}"
            )


# ===========================================================================
# TEST 3 — Single-layer update
# ===========================================================================

class TestSingleLayerUpdate:
    """Single-layer models must work identically to multi-layer."""

    def test_single_layer_all_keys_present(self, extractor: FeatureExtractor):
        params = {"only_layer": np.random.default_rng(1).normal(0, 1, (100,))}
        update = _make_update(params, update_id="upd-single")
        result = extractor.extract(update)
        missing = _ALL_REQUIRED_KEYS - result.keys()
        assert not missing, f"Missing keys: {missing}"

    def test_single_layer_layer_magnitudes_has_one_key(self, extractor: FeatureExtractor):
        params = {"only_layer": np.ones((50,), dtype=np.float64)}
        update = _make_update(params, update_id="upd-single-2")
        result = extractor.extract(update)
        assert list(result["layer_magnitudes"].keys()) == ["only_layer"]

    def test_single_layer_magnitude_correct(self, extractor: FeatureExtractor):
        arr = np.array([3.0, 4.0], dtype=np.float64)
        params = {"vec": arr}
        update = _make_update(params, update_id="upd-single-3")
        result = extractor.extract(update)
        assert abs(result["l2_norm"] - 5.0) < 1e-9
        assert abs(result["layer_magnitudes"]["vec"] - 5.0) < 1e-9


# ===========================================================================
# TEST 4 — Empty parameters
# ===========================================================================

class TestEmptyParameters:
    """Empty parameters dict must return zeros, not crash."""

    def test_empty_params_no_exception(self, extractor: FeatureExtractor):
        update = _make_update({}, update_id="upd-empty")
        result = extractor.extract(update)  # must not raise
        assert result is not None

    def test_empty_params_numeric_features_are_zero(self, extractor: FeatureExtractor):
        update = _make_update({}, update_id="upd-empty-2")
        result = extractor.extract(update)
        for key in ["l2_norm", "mean", "std", "max_abs", "min_abs",
                    "update_magnitude", "parameter_sparsity"]:
            assert result[key] == 0.0, f"Expected 0.0 for '{key}', got {result[key]}"

    def test_empty_params_layer_magnitudes_is_empty_dict(self, extractor: FeatureExtractor):
        update = _make_update({}, update_id="upd-empty-3")
        result = extractor.extract(update)
        assert result["layer_magnitudes"] == {}

    def test_empty_params_metadata_preserved(self, extractor: FeatureExtractor):
        update = _make_update({}, update_id="upd-empty-meta", client_id="c99", round_id=7)
        result = extractor.extract(update)
        assert result["update_id"] == "upd-empty-meta"
        assert result["client_id"] == "c99"
        assert result["round_id"] == 7


# ===========================================================================
# TEST 5 — NaN parameters
# ===========================================================================

class TestNaNParameters:
    """NaN values must raise FeatureExtractionError."""

    def test_nan_raises_feature_extraction_error(self, extractor: FeatureExtractor):
        params = {"layer": np.array([1.0, float("nan"), 3.0])}
        update = _make_update(params, update_id="upd-nan")
        with pytest.raises(FeatureExtractionError) as exc_info:
            extractor.extract(update)
        assert "NaN" in str(exc_info.value)
        assert "upd-nan" in str(exc_info.value)

    def test_nan_error_is_exception_subclass(self, extractor: FeatureExtractor):
        params = {"layer": np.array([float("nan")])}
        update = _make_update(params, update_id="upd-nan-2")
        with pytest.raises(Exception):  # Must be catchable as Exception
            extractor.extract(update)

    def test_all_nan_raises(self, extractor: FeatureExtractor):
        params = {"layer": np.full((10, 10), float("nan"))}
        update = _make_update(params, update_id="upd-all-nan")
        with pytest.raises(FeatureExtractionError):
            extractor.extract(update)

    def test_nan_in_second_layer_raises(self, extractor: FeatureExtractor):
        params = {
            "clean_layer": np.ones((10,)),
            "nan_layer": np.array([1.0, float("nan")]),
        }
        update = _make_update(params, update_id="upd-nan-second")
        with pytest.raises(FeatureExtractionError) as exc_info:
            extractor.extract(update)
        assert "NaN" in str(exc_info.value)


# ===========================================================================
# TEST 6 — Inf parameters
# ===========================================================================

class TestInfParameters:
    """Inf values must raise FeatureExtractionError."""

    def test_positive_inf_raises(self, extractor: FeatureExtractor):
        params = {"layer": np.array([1.0, float("inf"), 3.0])}
        update = _make_update(params, update_id="upd-inf")
        with pytest.raises(FeatureExtractionError) as exc_info:
            extractor.extract(update)
        assert "Inf" in str(exc_info.value)
        assert "upd-inf" in str(exc_info.value)

    def test_negative_inf_raises(self, extractor: FeatureExtractor):
        params = {"layer": np.array([1.0, float("-inf"), 3.0])}
        update = _make_update(params, update_id="upd-neg-inf")
        with pytest.raises(FeatureExtractionError):
            extractor.extract(update)

    def test_all_inf_raises(self, extractor: FeatureExtractor):
        params = {"layer": np.full((5,), float("inf"))}
        update = _make_update(params, update_id="upd-all-inf")
        with pytest.raises(FeatureExtractionError):
            extractor.extract(update)

    def test_inf_error_is_exception_subclass(self, extractor: FeatureExtractor):
        params = {"layer": np.array([float("inf")])}
        update = _make_update(params, update_id="upd-inf-exc")
        with pytest.raises(Exception):
            extractor.extract(update)


# ===========================================================================
# TEST 7 — Zero sample count
# ===========================================================================

class TestZeroSampleCount:
    """sample_count=0 is valid metadata; feature math must still work."""

    def test_zero_sample_count_no_exception(self, extractor: FeatureExtractor):
        params = {"layer": np.array([0.1, 0.2, 0.3])}
        update = _make_update(params, update_id="upd-zero-samples", sample_count=0)
        result = extractor.extract(update)
        assert result is not None

    def test_zero_sample_count_metadata_preserved(self, extractor: FeatureExtractor):
        params = {"layer": np.array([0.1, 0.2, 0.3])}
        update = _make_update(params, update_id="upd-zero-samples-2", sample_count=0)
        result = extractor.extract(update)
        assert result["sample_count"] == 0

    def test_zero_sample_count_features_finite(self, extractor: FeatureExtractor):
        params = {"layer": np.array([0.1, 0.2, 0.3])}
        update = _make_update(params, update_id="upd-zero-samples-3", sample_count=0)
        result = extractor.extract(update)
        for key in ["l2_norm", "mean", "std", "max_abs", "min_abs",
                    "update_magnitude", "parameter_sparsity"]:
            assert math.isfinite(result[key])

    def test_zero_sample_count_none_metadata(self, extractor: FeatureExtractor):
        """None values for optional metadata fields must also be preserved."""
        params = {"layer": np.array([0.5])}
        update = _make_update(
            params, update_id="upd-none-meta", sample_count=0,
            learning_rate=None, local_loss=None, local_accuracy=None,
        )
        result = extractor.extract(update)
        assert result["learning_rate"] is None
        assert result["local_loss"] is None
        assert result["local_accuracy"] is None


# ===========================================================================
# TEST 8 — Very large tensor
# ===========================================================================

class TestVeryLargeTensor:
    """10M parameter tensor must complete without crash or memory error."""

    def test_large_tensor_completes(self, extractor: FeatureExtractor):
        rng = np.random.default_rng(0)
        large = rng.normal(0, 0.01, (10_000_000,))
        params = {"huge_layer": large}
        update = _make_update(params, update_id="upd-large")
        result = extractor.extract(update)
        assert result is not None

    def test_large_tensor_all_keys_present(self, extractor: FeatureExtractor):
        rng = np.random.default_rng(1)
        large = rng.normal(0, 0.01, (10_000_000,))
        params = {"huge_layer": large}
        update = _make_update(params, update_id="upd-large-keys")
        result = extractor.extract(update)
        missing = _ALL_REQUIRED_KEYS - result.keys()
        assert not missing

    def test_large_tensor_features_finite(self, extractor: FeatureExtractor):
        rng = np.random.default_rng(2)
        large = rng.normal(0, 0.01, (10_000_000,))
        params = {"huge_layer": large}
        update = _make_update(params, update_id="upd-large-finite")
        result = extractor.extract(update)
        for key in ["l2_norm", "mean", "std", "max_abs", "min_abs",
                    "update_magnitude", "parameter_sparsity"]:
            assert math.isfinite(result[key])


# ===========================================================================
# QUALITY CHECKS
# ===========================================================================

class TestQualityChecks:
    """Non-functional correctness and quality guarantees."""

    def test_version_constant_exists_and_nonempty(self):
        assert isinstance(FEATURE_EXTRACTOR_VERSION, str)
        assert len(FEATURE_EXTRACTOR_VERSION) > 0

    def test_version_constant_value(self):
        assert FEATURE_EXTRACTOR_VERSION == "feature-extractor-v1"

    def test_schema_version_constant_exists_and_nonempty(self):
        assert isinstance(SCHEMA_VERSION, str)
        assert len(SCHEMA_VERSION) > 0

    def test_schema_version_constant_value(self):
        # Phase 1 QA: corrected from "feature-schema-v1" to "schema-v1" per contract.
        assert SCHEMA_VERSION == "schema-v1"

    def test_schema_version_in_output_dict(self):
        """Output dict must contain schema_version for Phase 2 consumers."""
        extractor = FeatureExtractor()
        params = {"layer": np.array([1.0, 2.0, 3.0])}
        update = _make_update(params, update_id="upd-schema-ver")
        result = extractor.extract(update)
        assert "schema_version" in result, "schema_version key missing from output"
        assert result["schema_version"] == SCHEMA_VERSION

    def test_feature_extraction_error_is_exception_subclass(self):
        assert issubclass(FeatureExtractionError, Exception)

    def test_feature_extraction_error_can_be_raised_and_caught(self):
        with pytest.raises(FeatureExtractionError):
            raise FeatureExtractionError("test error")

    def test_feature_extraction_error_message_preserved(self):
        msg = "unique test message 42"
        with pytest.raises(FeatureExtractionError, match=msg):
            raise FeatureExtractionError(msg)

    def test_deterministic_output(self):
        """Two calls with the same update must produce identical output."""
        rng = np.random.default_rng(55)
        params = {"w": rng.normal(0, 1, (20, 20)), "b": rng.normal(0, 1, (20,))}
        update = _make_update(params, update_id="upd-det")
        extractor = FeatureExtractor()
        result_a = extractor.extract(update)
        result_b = extractor.extract(update)
        # Compare numeric features
        for key in ["l2_norm", "mean", "std", "max_abs", "min_abs",
                    "update_magnitude", "parameter_sparsity"]:
            assert result_a[key] == result_b[key], f"Non-deterministic feature: '{key}'"
        assert result_a["layer_magnitudes"] == result_b["layer_magnitudes"]

    def test_no_mutation_of_input_parameters(self):
        """FeatureExtractor must NOT mutate the input parameters dict or tensors."""
        original_arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        params = {"layer": original_arr.copy()}
        original_copy = copy.deepcopy(params)
        update = _make_update(params, update_id="upd-no-mutate")
        extractor = FeatureExtractor()
        extractor.extract(update)
        # Verify the dict is unchanged
        assert set(update.parameters.keys()) == set(original_copy.keys())
        np.testing.assert_array_equal(
            np.array(update.parameters["layer"]),
            np.array(original_copy["layer"]),
        )

    def test_no_mutation_of_input_update_object(self):
        """The update object's other attributes must not be modified."""
        params = {"layer": np.array([0.5, 0.6])}
        update = _make_update(params, update_id="upd-no-mutate-obj",
                              client_id="orig-client", round_id=42)
        extractor = FeatureExtractor()
        extractor.extract(update)
        assert update.client_id == "orig-client"
        assert update.round_id == 42
        assert update.update_id == "upd-no-mutate-obj"

    def test_docstring_on_feature_extractor_class(self):
        assert FeatureExtractor.__doc__ is not None
        assert len(FeatureExtractor.__doc__.strip()) > 0

    def test_docstring_on_extract_method(self):
        assert FeatureExtractor.extract.__doc__ is not None
        assert len(FeatureExtractor.extract.__doc__.strip()) > 0

    def test_docstring_on_sentinel_class(self):
        from core.sentinel.sentinel import Sentinel
        assert Sentinel.__doc__ is not None
        assert len(Sentinel.__doc__.strip()) > 0

    def test_logging_fires_on_extract(self):
        """Logger must emit at least one record during a successful extract."""
        extractor = FeatureExtractor()
        params = {"layer": np.array([1.0, 2.0])}
        update = _make_update(params, update_id="upd-log-test")

        log_records: list[logging.LogRecord] = []

        class CapturingHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                log_records.append(record)

        handler = CapturingHandler()
        handler.setLevel(logging.DEBUG)

        # Attach to the feature_extractor logger and root to catch all
        fe_logger = logging.getLogger("core.sentinel.feature_extractor")
        fe_logger.addHandler(handler)
        fe_logger.setLevel(logging.DEBUG)

        try:
            extractor.extract(update)
        finally:
            fe_logger.removeHandler(handler)

        assert len(log_records) > 0, (
            "Expected at least one log record from FeatureExtractor.extract, got none."
        )

    def test_missing_attribute_raises_feature_extraction_error(self):
        """An object missing required attributes must raise FeatureExtractionError."""
        class BadUpdate:
            update_id = "bad-upd"
            # client_id intentionally missing

        extractor = FeatureExtractor()
        with pytest.raises(FeatureExtractionError) as exc_info:
            extractor.extract(BadUpdate())
        assert "client_id" in str(exc_info.value) or "missing" in str(exc_info.value).lower()

    def test_parameters_not_dict_raises(self):
        """parameters must be a dict; anything else raises FeatureExtractionError."""
        update = _make_update({"layer": np.array([1.0])}, update_id="upd-param-type")
        update.parameters = [1.0, 2.0]  # Wrong type: list instead of dict
        extractor = FeatureExtractor()
        with pytest.raises(FeatureExtractionError):
            extractor.extract(update)

    # ------------------------------------------------------------------
    # Phase 1 QA additions
    # ------------------------------------------------------------------

    def test_unsupported_parameter_type_string_raises(self):
        """A string value must raise FeatureExtractionError with 'Unsupported' in message."""
        update = _make_update(
            {"layer1": "hello"},
            update_id="upd-str-type",
        )
        extractor = FeatureExtractor()
        with pytest.raises(FeatureExtractionError) as exc_info:
            extractor.extract(update)
        msg = str(exc_info.value)
        assert "Unsupported" in msg or "unsupported" in msg.lower(), (
            f"Expected 'Unsupported' in error message, got: {msg!r}"
        )
        assert "layer1" in msg

    def test_unsupported_parameter_type_dict_raises(self):
        """A nested dict value must raise FeatureExtractionError."""
        update = _make_update(
            {"layer1": {"nested": 1.0}},
            update_id="upd-dict-type",
        )
        extractor = FeatureExtractor()
        with pytest.raises(FeatureExtractionError):
            extractor.extract(update)

    def test_layer_shapes_present_in_output(self):
        """Output must contain 'layer_shapes' key."""
        params = {
            "w": np.random.default_rng(0).normal(0, 1, (4, 8)),
            "b": np.random.default_rng(1).normal(0, 1, (4,)),
        }
        extractor = FeatureExtractor()
        result = extractor.extract(_make_update(params, update_id="upd-shapes"))
        assert "layer_shapes" in result
        assert isinstance(result["layer_shapes"], dict)

    def test_layer_shapes_values_correct(self):
        """layer_shapes must record the original tensor shape before flattening."""
        params = {
            "conv.weight": np.ones((32, 3, 3, 3), dtype=np.float32),
            "conv.bias": np.ones((32,), dtype=np.float32),
        }
        extractor = FeatureExtractor()
        result = extractor.extract(_make_update(params, update_id="upd-shapes-2"))
        assert result["layer_shapes"]["conv.weight"] == (32, 3, 3, 3)
        assert result["layer_shapes"]["conv.bias"] == (32,)

    def test_flat_parameters_present_and_correct(self):
        """flat_parameters must be a 1-D float64 array of all concatenated params."""
        arr_a = np.array([1.0, 2.0, 3.0])
        arr_b = np.array([4.0, 5.0])
        params = {"a": arr_a, "b": arr_b}
        extractor = FeatureExtractor()
        result = extractor.extract(_make_update(params, update_id="upd-flat"))
        assert "flat_parameters" in result
        flat = result["flat_parameters"]
        assert isinstance(flat, np.ndarray)
        assert flat.ndim == 1
        assert flat.dtype == np.float64
        assert flat.size == 5
        # Values must match concatenation (dict order preserved in Python 3.7+)
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_almost_equal(flat, expected)

    def test_layer_parameters_present_and_correct(self):
        """layer_parameters must be a dict of per-layer 1-D float64 arrays."""
        params = {
            "w": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "b": np.array([5.0, 6.0]),
        }
        extractor = FeatureExtractor()
        result = extractor.extract(_make_update(params, update_id="upd-layer-params"))
        assert "layer_parameters" in result
        lp = result["layer_parameters"]
        assert isinstance(lp, dict)
        assert set(lp.keys()) == {"w", "b"}
        np.testing.assert_array_almost_equal(lp["w"], np.array([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_array_almost_equal(lp["b"], np.array([5.0, 6.0]))
