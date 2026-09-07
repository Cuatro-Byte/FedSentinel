"""
core/sentinel/statistics.py

FedSentinel — Statistics Engine
Owner: Person 3 — FedSentinel Detection, Impact & Recovery Intelligence

Phase 2 — Descriptive Statistics Analysis.

This module receives the feature dictionary produced by FeatureExtractor
and computes descriptive statistics required by the downstream similarity,
anomaly detection, and threat scoring modules.

This module performs ANALYSIS ONLY.
It MUST NOT:
  - classify updates as malicious or benign;
  - compute anomaly scores, threat scores, similarity scores, reputation
    scores, impact scores, or response decisions;
  - perform any aggregation-level logic;
  - access raw client training data.

Contract reference: FEDSENTINEL_NEW_TEAM_ENGINEERING_CONTRACT_v2.md §13–14
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Version constants — required by Engineering Contract v2.0
# ---------------------------------------------------------------------------
STATISTICS_ENGINE_VERSION: str = "statistics-v1"

# Must match the schema version produced by FeatureExtractor.
# If the extractor schema changes, update this and bump STATISTICS_ENGINE_VERSION.
SCHEMA_VERSION: str = "schema-v1"

# Zero threshold for sparsity calculation — kept consistent with FeatureExtractor.
_SPARSITY_ZERO_THRESHOLD: float = 1e-6

# Minimum std below which skewness/kurtosis are defined as 0.0 (degenerate dist).
_STD_DEGENERATE_THRESHOLD: float = 1e-10

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Exception
# ---------------------------------------------------------------------------

class StatisticsEngineError(Exception):
    """
    Raised when the Statistics Engine cannot safely compute statistics.

    Scenarios that trigger this exception:
    - The features dictionary is missing required keys.
    - The features dictionary contains NaN or Inf in ``flat_parameters``.
    - Any other unrecoverable statistical computation failure.
    """


# ---------------------------------------------------------------------------
# StatisticsEngine
# ---------------------------------------------------------------------------

class StatisticsEngine:
    """
    Computes descriptive statistics from a FeatureExtractor output dictionary.

    The engine consumes the feature dict produced by
    :class:`~core.sentinel.feature_extractor.FeatureExtractor` and returns a
    statistics dictionary used by downstream detection modules.

    Statistics groups
    -----------------
    - ``global_statistics``       : variance, median, MAD, min, max, range,
                                    skewness, excess kurtosis.
    - ``distribution_statistics`` : percentile_25/50/75, IQR.
    - ``magnitude_statistics``    : mean_absolute_value, max_absolute_value,
                                    root_mean_square, energy.
    - ``sparsity_statistics``     : parameter_sparsity, non_zero_ratio,
                                    zero_count, non_zero_count.
    - ``layer_statistics``        : per-layer sub-dict with 12 scalar fields.

    This engine performs ANALYSIS ONLY. It does NOT output anomaly scores,
    threat scores, similarity scores, reputation, impact scores, or actions.

    Edge-case guarantees
    ---------------------
    - Empty ``flat_parameters`` (no layers): all scalar stats are 0.0,
      ``layer_statistics`` is ``{}``.
    - Constant tensor: ``std=0``, ``skewness=0``, ``kurtosis=0``.
    - Single-value tensor: ``std=0``, valid mean/median.
    - All-zero tensor: ``sparsity=1.0``, all magnitude stats are 0.0.
    - Sparse tensor: correct sparsity ratio, correct non_zero counts.
    - NaN / Inf in ``flat_parameters``: raises :class:`StatisticsEngineError`.
    - Negative values: handled; ``min_abs`` reflects absolute minimum correctly.
    - Mixed pos/neg: correct mean, std, skewness computation.

    Examples
    --------
    >>> engine = StatisticsEngine()
    >>> stats = engine.compute(features)
    >>> stats["global_statistics"]["median"]
    0.0012
    >>> stats["layer_statistics"]["conv1.weight"]["energy"]
    3.2145
    """

    def __init__(self) -> None:
        """Initialise the StatisticsEngine (stateless — no configuration needed)."""
        self._logger = logging.getLogger(
            self.__class__.__module__ + "." + self.__class__.__name__
        )
        self._logger.debug(
            "StatisticsEngine initialised (version=%s schema=%s)",
            STATISTICS_ENGINE_VERSION,
            SCHEMA_VERSION,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, features: dict[str, Any]) -> dict[str, Any]:
        """
        Compute descriptive statistics from a FeatureExtractor output dictionary.

        Parameters
        ----------
        features:
            The dictionary returned by
            :meth:`~core.sentinel.feature_extractor.FeatureExtractor.extract`.
            Must contain ``update_id``, ``client_id``, ``round_id``,
            ``flat_parameters``, and ``layer_parameters``.

        Returns
        -------
        dict[str, Any]
            Statistics dictionary with keys:
            ``update_id``, ``client_id``, ``round_id``,
            ``schema_version``, ``statistics_version``,
            ``global_statistics``, ``distribution_statistics``,
            ``magnitude_statistics``, ``sparsity_statistics``,
            ``layer_statistics``.

        Raises
        ------
        StatisticsEngineError
            If required keys are missing, or if NaN/Inf values are detected
            in ``flat_parameters``.
        """
        # --- Validate input ----------------------------------------------
        update_id = self._validate_features(features)

        self._logger.debug(
            "Computing statistics for update_id=%s", update_id
        )

        flat: np.ndarray = features.get(
            "flat_parameters", np.array([], dtype=np.float64)
        )
        layer_params: dict[str, np.ndarray] = features.get("layer_parameters", {})

        # Ensure flat_parameters is a numpy array
        if not isinstance(flat, np.ndarray):
            flat = np.asarray(flat, dtype=np.float64)

        # --- Compute statistic groups ------------------------------------
        global_stats = self._compute_global_statistics(flat, update_id)
        dist_stats = self._compute_distribution_statistics(flat, update_id)
        mag_stats = self._compute_magnitude_statistics(flat, update_id)
        sparsity_stats = self._compute_sparsity_statistics(flat, update_id)
        layer_stats = self._compute_layer_statistics(layer_params, update_id)

        stats_dict: dict[str, Any] = {
            # --- Identity ---
            "update_id": features["update_id"],
            "client_id": features["client_id"],
            "round_id": features["round_id"],
            # --- Versioning ---
            "schema_version": SCHEMA_VERSION,
            "statistics_version": STATISTICS_ENGINE_VERSION,
            # --- Statistics groups ---
            "global_statistics": global_stats,
            "distribution_statistics": dist_stats,
            "magnitude_statistics": mag_stats,
            "sparsity_statistics": sparsity_stats,
            "layer_statistics": layer_stats,
        }

        self._logger.info(
            "Statistics computed: update_id=%s n_params=%d "
            "median=%.6f mad=%.6f skewness=%.4f kurtosis=%.4f sparsity=%.4f",
            update_id,
            flat.size,
            global_stats["median"],
            global_stats["mad"],
            global_stats["skewness"],
            global_stats["kurtosis"],
            sparsity_stats["parameter_sparsity"],
        )

        return stats_dict

    # ------------------------------------------------------------------
    # Private: input validation
    # ------------------------------------------------------------------

    def _validate_features(self, features: dict[str, Any]) -> str:
        """
        Validate the features dictionary before computing statistics.

        Parameters
        ----------
        features:
            Input feature dictionary.

        Returns
        -------
        str
            The ``update_id`` extracted from ``features``.

        Raises
        ------
        StatisticsEngineError
            If required keys are absent, or if ``flat_parameters`` contains
            NaN or Inf values.
        """
        if not isinstance(features, dict):
            raise StatisticsEngineError(
                f"features must be a dict, got {type(features).__name__}."
            )

        required_keys = {"update_id", "client_id", "round_id"}
        missing = required_keys - features.keys()
        if missing:
            raise StatisticsEngineError(
                f"Feature dictionary is missing required key(s): {sorted(missing)}. "
                "Ensure the feature dict was produced by FeatureExtractor."
            )

        update_id: str = str(features["update_id"])

        # Schema version advisory — warn but do not crash
        schema = features.get("schema_version", "<absent>")
        if schema != SCHEMA_VERSION:
            self._logger.warning(
                "Feature schema version mismatch: expected='%s' got='%s' "
                "for update_id=%s. Statistics may be inaccurate.",
                SCHEMA_VERSION,
                schema,
                update_id,
            )

        # NaN / Inf check on flat_parameters
        flat = features.get("flat_parameters")
        if flat is not None:
            if not isinstance(flat, np.ndarray):
                flat = np.asarray(flat, dtype=np.float64)
            if flat.size > 0:
                nan_count = int(np.sum(np.isnan(flat)))
                inf_count = int(np.sum(np.isinf(flat)))
                if nan_count > 0:
                    raise StatisticsEngineError(
                        f"flat_parameters for update_id={update_id} contains "
                        f"{nan_count} NaN value(s). Cannot compute statistics on "
                        "corrupted data."
                    )
                if inf_count > 0:
                    raise StatisticsEngineError(
                        f"flat_parameters for update_id={update_id} contains "
                        f"{inf_count} Inf value(s). Cannot compute statistics on "
                        "corrupted data."
                    )

        return update_id

    # ------------------------------------------------------------------
    # Private: statistic group helpers (all pure — no side effects)
    # ------------------------------------------------------------------

    def _compute_global_statistics(
        self, flat: np.ndarray, update_id: str
    ) -> dict[str, float]:
        """
        Compute global descriptive statistics over all flattened parameters.

        Parameters
        ----------
        flat:
            Sanitised 1-D float64 array of all parameters.
        update_id:
            Used for error messages.

        Returns
        -------
        dict[str, float]
            Keys: ``variance``, ``median``, ``mad``, ``minimum``,
            ``maximum``, ``value_range``, ``skewness``, ``kurtosis``.
        """
        _zero = {
            "variance": 0.0,
            "median": 0.0,
            "mad": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
            "value_range": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
        }

        if flat.size == 0:
            return _zero

        if flat.size == 1:
            v = float(flat[0])
            return {
                "variance": 0.0,
                "median": v,
                "mad": 0.0,
                "minimum": v,
                "maximum": v,
                "value_range": 0.0,
                "skewness": 0.0,
                "kurtosis": 0.0,
            }

        variance = float(np.var(flat))
        median = float(np.median(flat))
        mad = float(np.median(np.abs(flat - median)))
        minimum = float(np.min(flat))
        maximum = float(np.max(flat))
        value_range = maximum - minimum

        # Skewness and excess kurtosis — degenerate (zero std) → defined as 0.0
        std = math.sqrt(variance) if variance > 0.0 else 0.0
        if std < _STD_DEGENERATE_THRESHOLD:
            skewness = 0.0
            kurtosis = 0.0
        else:
            centered = flat - float(np.mean(flat))
            skewness = float(np.mean(centered ** 3) / std ** 3)
            kurtosis = float(np.mean(centered ** 4) / std ** 4 - 3.0)  # excess

        result = {
            "variance": variance,
            "median": median,
            "mad": mad,
            "minimum": minimum,
            "maximum": maximum,
            "value_range": value_range,
            "skewness": skewness,
            "kurtosis": kurtosis,
        }

        # Final finite check
        for key, val in result.items():
            if not math.isfinite(val):
                raise StatisticsEngineError(
                    f"Global statistic '{key}' is not finite ({val}) "
                    f"for update_id={update_id}."
                )

        return result

    def _compute_distribution_statistics(
        self, flat: np.ndarray, update_id: str
    ) -> dict[str, float]:
        """
        Compute percentile and IQR distribution statistics.

        Parameters
        ----------
        flat:
            Sanitised 1-D float64 array.
        update_id:
            Used for error messages.

        Returns
        -------
        dict[str, float]
            Keys: ``percentile_25``, ``percentile_50``, ``percentile_75``,
            ``interquartile_range``.
        """
        if flat.size == 0:
            return {
                "percentile_25": 0.0,
                "percentile_50": 0.0,
                "percentile_75": 0.0,
                "interquartile_range": 0.0,
            }

        # Compute all three in a single numpy call (single sort pass internally)
        p25, p50, p75 = (
            float(x) for x in np.percentile(flat, [25.0, 50.0, 75.0])
        )
        iqr = p75 - p25

        return {
            "percentile_25": p25,
            "percentile_50": p50,
            "percentile_75": p75,
            "interquartile_range": iqr,
        }

    def _compute_magnitude_statistics(
        self, flat: np.ndarray, update_id: str
    ) -> dict[str, float]:
        """
        Compute magnitude-based statistics (absolute and squared norms).

        Parameters
        ----------
        flat:
            Sanitised 1-D float64 array.
        update_id:
            Used for error messages.

        Returns
        -------
        dict[str, float]
            Keys: ``mean_absolute_value``, ``max_absolute_value``,
            ``root_mean_square``, ``energy``.
        """
        if flat.size == 0:
            return {
                "mean_absolute_value": 0.0,
                "max_absolute_value": 0.0,
                "root_mean_square": 0.0,
                "energy": 0.0,
            }

        abs_flat = np.abs(flat)
        squared = flat ** 2

        return {
            "mean_absolute_value": float(np.mean(abs_flat)),
            "max_absolute_value": float(np.max(abs_flat)),
            "root_mean_square": float(np.sqrt(np.mean(squared))),
            "energy": float(np.sum(squared)),
        }

    def _compute_sparsity_statistics(
        self, flat: np.ndarray, update_id: str
    ) -> dict[str, Any]:
        """
        Compute sparsity statistics.

        Parameters
        ----------
        flat:
            Sanitised 1-D float64 array.
        update_id:
            Used for error messages.

        Returns
        -------
        dict[str, Any]
            Keys: ``parameter_sparsity`` (float), ``non_zero_ratio`` (float),
            ``zero_count`` (int), ``non_zero_count`` (int).
        """
        if flat.size == 0:
            return {
                "parameter_sparsity": 0.0,
                "non_zero_ratio": 0.0,
                "zero_count": 0,
                "non_zero_count": 0,
            }

        near_zero_mask = np.abs(flat) < _SPARSITY_ZERO_THRESHOLD
        zero_count = int(np.sum(near_zero_mask))
        non_zero_count = flat.size - zero_count
        sparsity = zero_count / flat.size
        non_zero_ratio = non_zero_count / flat.size

        return {
            "parameter_sparsity": float(sparsity),
            "non_zero_ratio": float(non_zero_ratio),
            "zero_count": zero_count,
            "non_zero_count": non_zero_count,
        }

    def _compute_layer_statistics(
        self,
        layer_parameters: dict[str, np.ndarray],
        update_id: str,
    ) -> dict[str, dict[str, Any]]:
        """
        Compute per-layer descriptive statistics.

        For each layer produces a sub-dict with 12 fields:
        ``parameter_count``, ``mean``, ``std``, ``variance``, ``median``,
        ``mad``, ``l2_norm``, ``max_abs``, ``min_abs``, ``rms``,
        ``energy``, ``sparsity``.

        Parameters
        ----------
        layer_parameters:
            Dict mapping layer name → sanitised 1-D float64 array.
        update_id:
            Used for error messages and logging.

        Returns
        -------
        dict[str, dict[str, Any]]
            Outer keys: layer names. Inner keys: the 12 statistic fields.
        """
        result: dict[str, dict[str, Any]] = {}

        for layer_name, flat_layer in layer_parameters.items():
            if flat_layer.size == 0:
                result[layer_name] = {
                    "parameter_count": 0,
                    "mean": 0.0,
                    "std": 0.0,
                    "variance": 0.0,
                    "median": 0.0,
                    "mad": 0.0,
                    "l2_norm": 0.0,
                    "max_abs": 0.0,
                    "min_abs": 0.0,
                    "rms": 0.0,
                    "energy": 0.0,
                    "sparsity": 0.0,
                }
                continue

            layer_mean = float(np.mean(flat_layer))
            layer_std = float(np.std(flat_layer))
            layer_var = float(np.var(flat_layer))
            layer_median = float(np.median(flat_layer))
            layer_mad = float(np.median(np.abs(flat_layer - layer_median)))

            abs_layer = np.abs(flat_layer)
            layer_l2 = float(np.linalg.norm(flat_layer))
            layer_max_abs = float(np.max(abs_layer))
            layer_min_abs = float(np.min(abs_layer))
            layer_sq = flat_layer ** 2
            layer_rms = float(np.sqrt(np.mean(layer_sq)))
            layer_energy = float(np.sum(layer_sq))

            near_zero = np.sum(abs_layer < _SPARSITY_ZERO_THRESHOLD)
            layer_sparsity = float(near_zero / flat_layer.size)

            result[layer_name] = {
                "parameter_count": flat_layer.size,
                "mean": layer_mean,
                "std": layer_std,
                "variance": layer_var,
                "median": layer_median,
                "mad": layer_mad,
                "l2_norm": layer_l2,
                "max_abs": layer_max_abs,
                "min_abs": layer_min_abs,
                "rms": layer_rms,
                "energy": layer_energy,
                "sparsity": layer_sparsity,
            }

        return result
