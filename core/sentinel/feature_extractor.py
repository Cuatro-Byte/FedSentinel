"""
core/sentinel/feature_extractor.py

FedSentinel — Feature Extraction Engine
Owner: Person 3 — FedSentinel Detection, Impact & Recovery Intelligence

This module is the first stage of the FedSentinel detection pipeline.
It receives a canonical ModelUpdate object and extracts numerical features
required by downstream stages: statistics, similarity, anomaly detection,
threat scoring, and impact estimation.

Contract reference: FEDSENTINEL_NEW_TEAM_ENGINEERING_CONTRACT_v2.md §14

Phase 1 QA improvements applied (2026-09-07):
  - SCHEMA_VERSION corrected to "schema-v1".
  - Explicit parameter type whitelist; unsupported types raise FeatureExtractionError.
  - layer_shapes added to output for Impact Estimator.
  - flat_parameters and layer_parameters added for Statistics Engine (no tensor-handling
    duplication needed downstream).
  - Single-pass _process_parameters replaces two separate iterations.

Version: feature-extractor-v1
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Version constants — required by Engineering Contract v2.0
# ---------------------------------------------------------------------------
FEATURE_EXTRACTOR_VERSION: str = "feature-extractor-v1"

# Schema version of the output feature dictionary.
# Increment when the set or type of keys changes so downstream consumers
# (statistics engine, threat scorer, etc.) can detect schema mismatches.
# Phase 1 QA: corrected from "feature-schema-v1" → "schema-v1" per contract.
SCHEMA_VERSION: str = "schema-v1"

# Zero threshold for sparsity calculation
_SPARSITY_ZERO_THRESHOLD: float = 1e-6

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Exception
# ---------------------------------------------------------------------------

class FeatureExtractionError(Exception):
    """
    Raised when the feature extractor cannot safely process a ModelUpdate.

    Scenarios that trigger this exception:
    - The parameters dictionary contains an unsupported datatype.
    - The parameters dictionary contains NaN values.
    - The parameters dictionary contains Inf values.
    - The input object is missing required attributes.
    - Any other unrecoverable extraction failure.
    """


# ---------------------------------------------------------------------------
# FeatureExtractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Extracts numerical features from a canonical ModelUpdate object.

    This class is the sole entry point for Phase 1 feature extraction.
    It is stateless — every call to :meth:`extract` is independent and
    does not modify the input update.

    Extracted features (8 total)
    -----------------------------
    - ``l2_norm``            : L2 norm of all flattened parameters.
    - ``mean``               : Arithmetic mean of all parameters.
    - ``std``                : Standard deviation of all parameters.
    - ``max_abs``            : Maximum absolute value across all parameters.
    - ``min_abs``            : Minimum absolute value across all parameters.
    - ``update_magnitude``   : Root-mean-square (RMS) of all parameters.
    - ``parameter_sparsity`` : Fraction of parameters whose absolute value
                               is below ``_SPARSITY_ZERO_THRESHOLD`` (1e-6).
    - ``layer_magnitudes``   : Dict mapping layer name → L2 norm of that layer.

    Phase 1 QA additions (non-breaking)
    ------------------------------------
    - ``layer_shapes``       : Dict mapping layer name → original tensor shape tuple.
    - ``flat_parameters``    : Sanitised 1-D float64 array of all parameters (for
                               downstream Statistics Engine; no mutation).
    - ``layer_parameters``   : Dict mapping layer name → sanitised 1-D float64 array
                               (for downstream Statistics Engine; no mutation).

    Preserved metadata (8 fields)
    ------------------------------
    ``update_id``, ``client_id``, ``round_id``, ``sample_count``,
    ``training_epochs``, ``learning_rate``, ``local_loss``, ``local_accuracy``

    Supported parameter types
    --------------------------
    ``numpy.ndarray``, ``list``, ``tuple``, ``float``, ``int``,
    ``torch.Tensor`` (via ``.detach().numpy()`` or ``.numpy()``).
    All other types raise :class:`FeatureExtractionError`.

    Edge-case guarantees
    ---------------------
    - Empty ``parameters`` dict: all numeric features are 0.0;
      ``layer_magnitudes``, ``layer_shapes``, ``layer_parameters`` are empty dicts.
    - NaN in any tensor: raises :class:`FeatureExtractionError`.
    - Inf in any tensor: raises :class:`FeatureExtractionError`.
    - Unsupported type (e.g. str): raises :class:`FeatureExtractionError`.
    - Extremely large tensors: handled by NumPy without mutation.
    - Single-layer model: works identically to multi-layer.
    - Zero ``sample_count``: preserved as-is in metadata; not used in math.

    Examples
    --------
    >>> extractor = FeatureExtractor()
    >>> features = extractor.extract(update)
    >>> features["l2_norm"]
    0.5432
    >>> features["layer_shapes"]
    {"layer1.weight": (64, 32), "layer1.bias": (64,)}
    """

    def __init__(self) -> None:
        """Initialise the FeatureExtractor (stateless — no configuration needed)."""
        self._logger = logging.getLogger(
            self.__class__.__module__ + "." + self.__class__.__name__
        )
        self._logger.debug(
            "FeatureExtractor initialised (version=%s schema=%s)",
            FEATURE_EXTRACTOR_VERSION,
            SCHEMA_VERSION,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, update: Any) -> dict[str, Any]:
        """
        Extract features from a ModelUpdate and return a single dictionary.

        The returned dictionary contains:
        - 8 numerical feature fields.
        - 8 metadata fields preserving update provenance.
        - ``layer_shapes``      : original tensor shapes per layer.
        - ``flat_parameters``   : sanitised global flat array (for Statistics Engine).
        - ``layer_parameters``  : per-layer sanitised flat arrays (for Statistics Engine).
        - ``extractor_version`` : version constant.
        - ``schema_version``    : schema version constant (``"schema-v1"``).

        Parameters
        ----------
        update:
            A ``ModelUpdate`` instance conforming to the shared data contract
            (§8.1 of FEDSENTINEL_NEW_TEAM_ENGINEERING_CONTRACT_v2.md).
            Must expose: ``update_id``, ``client_id``, ``round_id``,
            ``parameters``, ``sample_count``, ``training_epochs``,
            ``learning_rate``, ``local_loss``, ``local_accuracy``.

        Returns
        -------
        dict[str, Any]
            Feature dictionary. See class docstring for the complete field list.

        Raises
        ------
        FeatureExtractionError
            If required attributes are missing, if a parameter layer has an
            unsupported type, or if NaN/Inf values are detected.
        """
        self._logger.debug(
            "Extracting features for update_id=%s client_id=%s round_id=%s",
            getattr(update, "update_id", "<unknown>"),
            getattr(update, "client_id", "<unknown>"),
            getattr(update, "round_id", "<unknown>"),
        )

        # --- Validate required attributes --------------------------------
        self._validate_update_attributes(update)

        update_id: str = update.update_id
        client_id: str = update.client_id
        round_id: int = update.round_id
        parameters: dict[str, Any] = update.parameters

        # --- Single-pass processing: flatten, sanitise, shapes, magnitudes
        flat, layer_magnitudes, layer_shapes, layer_parameters = (
            self._process_parameters(parameters, update_id)
        )

        # --- Compute global numerical features ---------------------------
        if flat.size == 0:
            l2_norm = 0.0
            mean = 0.0
            std = 0.0
            max_abs = 0.0
            min_abs = 0.0
            update_magnitude = 0.0
            parameter_sparsity = 0.0
        else:
            l2_norm = float(np.linalg.norm(flat))
            mean = float(np.mean(flat))
            std = float(np.std(flat))
            abs_flat = np.abs(flat)
            max_abs = float(np.max(abs_flat))
            min_abs = float(np.min(abs_flat))
            # RMS = sqrt(mean of squares)
            update_magnitude = float(np.sqrt(np.mean(flat ** 2)))
            near_zero = np.sum(abs_flat < _SPARSITY_ZERO_THRESHOLD)
            parameter_sparsity = float(near_zero / flat.size)

        # --- Validate computed values are finite -------------------------
        for name, value in [
            ("l2_norm", l2_norm),
            ("mean", mean),
            ("std", std),
            ("max_abs", max_abs),
            ("min_abs", min_abs),
            ("update_magnitude", update_magnitude),
            ("parameter_sparsity", parameter_sparsity),
        ]:
            if not math.isfinite(value):
                raise FeatureExtractionError(
                    f"Feature '{name}' is not finite ({value}) for update_id={update_id}. "
                    "This indicates an unrecoverable numerical issue in the parameters."
                )

        feature_dict: dict[str, Any] = {
            # --- Metadata (8 fields) ---
            "update_id": update_id,
            "client_id": client_id,
            "round_id": round_id,
            "sample_count": update.sample_count,
            "training_epochs": update.training_epochs,
            "learning_rate": update.learning_rate,
            "local_loss": update.local_loss,
            "local_accuracy": update.local_accuracy,
            # --- Features (8 fields) ---
            "l2_norm": l2_norm,
            "mean": mean,
            "std": std,
            "max_abs": max_abs,
            "min_abs": min_abs,
            "update_magnitude": update_magnitude,
            "parameter_sparsity": parameter_sparsity,
            "layer_magnitudes": layer_magnitudes,
            # --- Phase 1 QA additions ---
            "layer_shapes": layer_shapes,
            # --- Raw arrays for downstream consumers (Statistics Engine) ---
            "flat_parameters": flat,
            "layer_parameters": layer_parameters,
            # --- Provenance ---
            "extractor_version": FEATURE_EXTRACTOR_VERSION,
            "schema_version": SCHEMA_VERSION,
        }

        self._logger.info(
            "Features extracted: update_id=%s l2_norm=%.6f mean=%.6f std=%.6f "
            "max_abs=%.6f min_abs=%.6f update_magnitude=%.6f sparsity=%.4f "
            "layers=%d total_params=%d",
            update_id,
            l2_norm,
            mean,
            std,
            max_abs,
            min_abs,
            update_magnitude,
            parameter_sparsity,
            len(layer_magnitudes),
            flat.size,
        )

        return feature_dict

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_update_attributes(self, update: Any) -> None:
        """
        Verify the update object exposes all required attributes.

        Parameters
        ----------
        update:
            Any object expected to conform to the ModelUpdate contract.

        Raises
        ------
        FeatureExtractionError
            If any required attribute is missing or ``parameters`` is not a dict.
        """
        required = (
            "update_id",
            "client_id",
            "round_id",
            "parameters",
            "sample_count",
            "training_epochs",
            "learning_rate",
            "local_loss",
            "local_accuracy",
        )
        missing = [attr for attr in required if not hasattr(update, attr)]
        if missing:
            raise FeatureExtractionError(
                f"ModelUpdate is missing required attribute(s): {missing}. "
                "Ensure the update conforms to the shared data contract §8.1."
            )
        if not isinstance(update.parameters, dict):
            raise FeatureExtractionError(
                f"ModelUpdate.parameters must be a dict, got {type(update.parameters).__name__} "
                f"for update_id={getattr(update, 'update_id', '<unknown>')}."
            )

    def _process_parameters(
        self,
        parameters: dict[str, Any],
        update_id: str,
    ) -> tuple[np.ndarray, dict[str, float], dict[str, tuple], dict[str, np.ndarray]]:
        """
        Process all parameter layers in a single pass.

        Replaces the former ``_flatten_parameters`` + ``_compute_layer_magnitudes``
        pair. Iterates each layer exactly once, capturing:
        - the original tensor shape (before flattening),
        - the sanitised flat array (NaN/Inf checked),
        - the L2 norm per layer.

        Does NOT mutate the input dictionary or any tensor.

        Parameters
        ----------
        parameters:
            Layer-name → tensor mapping from ``ModelUpdate.parameters``.
        update_id:
            Used for error messages and logging.

        Returns
        -------
        flat : np.ndarray
            Sanitised 1-D float64 array of all parameters concatenated.
        layer_magnitudes : dict[str, float]
            Mapping of layer name → L2 norm (float).
        layer_shapes : dict[str, tuple]
            Mapping of layer name → original tensor shape tuple.
        layer_parameters : dict[str, np.ndarray]
            Mapping of layer name → sanitised 1-D float64 array.

        Raises
        ------
        FeatureExtractionError
            If any tensor has an unsupported type or contains NaN/Inf.
        """
        if not parameters:
            self._logger.debug(
                "update_id=%s has empty parameters dict; returning empty arrays.",
                update_id,
            )
            return (
                np.array([], dtype=np.float64),
                {},
                {},
                {},
            )

        arrays: list[np.ndarray] = []
        layer_magnitudes: dict[str, float] = {}
        layer_shapes: dict[str, tuple] = {}
        layer_parameters: dict[str, np.ndarray] = {}

        for layer_name, tensor in parameters.items():
            try:
                arr: np.ndarray = self._to_numpy(tensor, layer_name, update_id)
                # Record original shape before flattening
                layer_shapes[layer_name] = tuple(arr.shape)
                flat_layer: np.ndarray = arr.ravel().astype(np.float64, copy=False)
                # Validate NaN / Inf per layer (catches errors early with per-layer context)
                flat_layer = self._sanitize_array(flat_layer, update_id)
                layer_magnitudes[layer_name] = float(np.linalg.norm(flat_layer))
                layer_parameters[layer_name] = flat_layer
                arrays.append(flat_layer)
            except FeatureExtractionError:
                raise
            except Exception as exc:
                raise FeatureExtractionError(
                    f"Cannot process layer '{layer_name}' for update_id={update_id}: {exc}"
                ) from exc

        global_flat = (
            np.concatenate(arrays) if arrays else np.array([], dtype=np.float64)
        )
        return global_flat, layer_magnitudes, layer_shapes, layer_parameters

    def _to_numpy(self, tensor: Any, layer_name: str, update_id: str) -> np.ndarray:
        """
        Convert a supported tensor-like value to a NumPy array without mutation.

        Supported types
        ---------------
        - ``numpy.ndarray``  → copied via ``tensor.copy()``.
        - ``list``, ``tuple``→ converted via ``np.array(..., dtype=float64)``.
        - PyTorch grad tensor (has ``.detach()``) → ``tensor.detach().numpy()``.
        - PyTorch / TF tensor (has ``.numpy()``) → ``tensor.numpy()``.
        - Python scalar (``float``, ``int``, NumPy scalar) → wrapped in 1-D array.

        All other types raise :class:`FeatureExtractionError` immediately.
        Strings and bytes are explicitly excluded from the scalar path.

        Parameters
        ----------
        tensor:
            The raw parameter value for a single layer.
        layer_name:
            Layer name (used in error messages).
        update_id:
            Update identifier (used in error messages).

        Returns
        -------
        np.ndarray
            Float64 NumPy array. Never shares memory with the original input.

        Raises
        ------
        FeatureExtractionError
            If the type is unsupported or if tensor-framework conversion fails.
        """
        # 1. NumPy ndarray — most common case in FL
        if isinstance(tensor, np.ndarray):
            return tensor.copy()

        # 2. Python list or tuple
        if isinstance(tensor, (list, tuple)):
            return np.array(tensor, dtype=np.float64)

        # 3. PyTorch tensor requiring grad (.detach() must come before .numpy())
        if hasattr(tensor, "detach"):
            try:
                return np.array(tensor.detach().numpy(), dtype=np.float64)
            except Exception as exc:
                raise FeatureExtractionError(
                    f"Layer '{layer_name}' PyTorch .detach().numpy() failed "
                    f"for update_id={update_id}: {exc}"
                ) from exc

        # 4. TensorFlow / generic tensor with .numpy()
        if hasattr(tensor, "numpy"):
            try:
                return np.array(tensor.numpy(), dtype=np.float64)
            except Exception as exc:
                raise FeatureExtractionError(
                    f"Layer '{layer_name}' .numpy() conversion failed "
                    f"for update_id={update_id}: {exc}"
                ) from exc

        # 5. Python scalar (float, int, numpy scalar) — strings/bytes excluded
        if np.isscalar(tensor) and not isinstance(tensor, (str, bytes)):
            return np.array([tensor], dtype=np.float64)

        # 6. Unsupported type — explicit rejection (no silent fallback)
        raise FeatureExtractionError(
            f"Unsupported parameter type for layer '{layer_name}': "
            f"'{type(tensor).__name__}'. "
            f"Supported types: numpy.ndarray, list, tuple, scalar (float/int), "
            f"torch.Tensor (via .detach().numpy() or .numpy()). "
            f"update_id={update_id}"
        )

    def _sanitize_array(self, arr: np.ndarray, update_id: str) -> np.ndarray:
        """
        Validate a flattened parameter array for NaN and Inf values.

        If any NaN or Inf values are found, :class:`FeatureExtractionError`
        is raised immediately with a descriptive message. The input array is
        never mutated.

        Parameters
        ----------
        arr:
            1-D float64 NumPy array of all parameters.
        update_id:
            Used for error messages and logging.

        Returns
        -------
        np.ndarray
            The same array (unmodified) if no NaN/Inf detected.

        Raises
        ------
        FeatureExtractionError
            If any NaN or Inf values are present.
        """
        if arr.size == 0:
            return arr

        nan_count = int(np.sum(np.isnan(arr)))
        inf_count = int(np.sum(np.isinf(arr)))

        if nan_count > 0:
            self._logger.error(
                "update_id=%s contains %d NaN parameter(s). "
                "Raising FeatureExtractionError.",
                update_id,
                nan_count,
            )
            raise FeatureExtractionError(
                f"ModelUpdate update_id={update_id} contains {nan_count} NaN value(s) "
                "in its parameters. Cannot safely extract features from a poisoned or "
                "corrupted update. The update should be quarantined."
            )

        if inf_count > 0:
            self._logger.error(
                "update_id=%s contains %d Inf parameter(s). "
                "Raising FeatureExtractionError.",
                update_id,
                inf_count,
            )
            raise FeatureExtractionError(
                f"ModelUpdate update_id={update_id} contains {inf_count} Inf value(s) "
                "in its parameters. Cannot safely extract features from a poisoned or "
                "corrupted update. The update should be quarantined."
            )

        return arr
