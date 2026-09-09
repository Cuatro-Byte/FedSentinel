"""
impact_estimator.py
===================
FedSentinel — Person B (Person 3): Validation-Aware Sentinel Recovery

This module provides ``RecoveryTriggerEngine``, the core class responsible for
consuming anomaly scores from the detection pipeline together with optional
validation-layer outputs, and deciding whether the global model requires
selective recovery.

Engineering Contract v2.0 compliance
--------------------------------------
Allowed imports only:
    typing, dataclasses, enum, datetime, collections

Must NOT:
  - import aggregator.py, server.py, backend/*, attack_manager.py
  - perform aggregation, checkpoint recovery, model evaluation
  - access attack ground truth or raw client datasets
  - implement validator logic (only consume its outputs)

Public API (backward-compatible)
----------------------------------
::

    estimator = RecoveryTriggerEngine()

    decision: RecoveryDecision = estimator.evaluate_recovery_need(
        anomaly_scores={"client_01": 0.92, "client_02": 0.31},
        val_metrics={"val_loss": 0.91, "val_acc": 0.62,
                     "loss_delta": 0.19, "acc_delta": -0.08},
        loss_spiked=True,
    )

The ``val_metrics`` and ``loss_spiked`` parameters are optional and default
to ``None`` / ``False`` so that callers that only supply anomaly scores
continue to work without modification.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional

from .recovery_models import RecoveryDecision, RecoveryState


# ---------------------------------------------------------------------------
# Module-level configuration constants
# ---------------------------------------------------------------------------

# When True, a validation loss spike (loss_spiked=True) always overrides the
# anomaly-only evaluation and immediately triggers recovery.
LOSS_SPIKE_PRIORITY: bool = True

# Anomaly score threshold.  Clients whose score >= this value in a given round
# are considered suspicious for that round.
ANOMALY_THRESHOLD: float = 0.80

# Number of recent rounds to inspect when identifying persistently suspicious
# clients and selecting the recovery round.
LOOKBACK_ROUNDS: int = 3


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# A single round's record stored in the internal history deque.
# Structure:
#   {
#       "round": <int>,
#       "scores": {"client_id": <float>, ...}
#   }
_RoundRecord = Dict


class RecoveryTriggerEngine:
    """
    Validation-aware recovery decision engine for the FedSentinel pipeline.

    Responsibilities
    ----------------
    * Maintain a lightweight rolling history of per-round anomaly scores.
    * On each call to ``evaluate_recovery_need``:
        1. Store the current round's scores in history.
        2. Apply Rule 1 — Validation Override (if ``loss_spiked`` is True).
        3. Apply Rule 2 — Anomaly-Based Trigger (if Rule 1 did not fire).
        4. Apply Rule 3 — Historical Client Analysis to identify quarantined
           clients and select the earliest affected recovery round.
    * Return a ``RecoveryDecision`` data carrier (does NOT execute recovery).

    Configuration
    -------------
    All thresholds are drawn from the module-level constants:
        ``ANOMALY_THRESHOLD``, ``LOOKBACK_ROUNDS``, ``LOSS_SPIKE_PRIORITY``.

    Parameters
    ----------
    anomaly_threshold : float, optional
        Override for ``ANOMALY_THRESHOLD``.  Must be in (0.0, 1.0].
    lookback_rounds : int, optional
        Override for ``LOOKBACK_ROUNDS``.  Must be >= 1.
    loss_spike_priority : bool, optional
        Override for ``LOSS_SPIKE_PRIORITY``.

    Attributes
    ----------
    current_round : int
        Auto-incrementing counter tracking how many evaluations have been
        performed.  Starts at 0 and advances by 1 on each call to
        ``evaluate_recovery_need``.
    recovery_state : RecoveryState
        The last recorded state.  Initialises to ``NOT_REQUIRED``.

    Notes
    -----
    This class does NOT load checkpoints, perform aggregation, or interact
    with any model.  It is a pure decision engine.
    """

    def __init__(
        self,
        anomaly_threshold: float = ANOMALY_THRESHOLD,
        lookback_rounds: int = LOOKBACK_ROUNDS,
        loss_spike_priority: bool = LOSS_SPIKE_PRIORITY,
    ) -> None:
        # Validate constructor arguments
        if not (0.0 < anomaly_threshold <= 1.0):
            raise ValueError(
                f"anomaly_threshold must be in (0.0, 1.0], got {anomaly_threshold}"
            )
        if lookback_rounds < 1:
            raise ValueError(
                f"lookback_rounds must be >= 1, got {lookback_rounds}"
            )

        # Configuration
        self._anomaly_threshold: float = anomaly_threshold
        self._lookback_rounds: int = lookback_rounds
        self._loss_spike_priority: bool = loss_spike_priority

        # State
        self.current_round: int = 0
        self.recovery_state: RecoveryState = RecoveryState.NOT_REQUIRED

        # Rolling history — oldest entries automatically evicted when the
        # deque reaches its maximum length.
        self._round_history: Deque[_RoundRecord] = deque(
            maxlen=lookback_rounds
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_recovery_need(
        self,
        anomaly_scores: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        loss_spiked: bool = False,
    ) -> RecoveryDecision:
        """
        Evaluate whether the current round requires selective recovery.

        Parameters
        ----------
        anomaly_scores : Dict[str, float]
            Mapping of ``client_id`` → anomaly score in [0.0, 1.0] for the
            current federated round.  Produced upstream by the anomaly
            detector / threat scorer.
        val_metrics : Optional[Dict[str, float]]
            Validation metrics from Person A's ServerValidator, structured as::

                {
                    "val_loss":   float,
                    "val_acc":    float,
                    "loss_delta": float,
                    "acc_delta":  float,
                }

            May be ``None`` when no validation checkpoint was evaluated.
            The estimator does NOT compute or modify these values.
        loss_spiked : bool
            ``True`` when Person A's validator detected a significant loss
            increase (``loss_spiked`` as returned by
            ``validator.evaluate_checkpoint(candidate_model)``).
            Defaults to ``False`` for backward compatibility.

        Returns
        -------
        RecoveryDecision
            A populated ``RecoveryDecision`` describing whether recovery is
            needed, which clients to quarantine, and the suggested recovery
            round.  Does NOT trigger recovery itself.

        Decision Rules (in priority order)
        ------------------------------------
        1. **Validation Override** — if ``loss_spiked is True`` and
           ``LOSS_SPIKE_PRIORITY`` is enabled, immediately set state to
           TRIGGERED with reason ``"VALIDATION_LOSS_SPIKE"``.
        2. **Anomaly-Based Trigger** — inspect all clients in
           ``anomaly_scores``; if any score >= ``ANOMALY_THRESHOLD``, trigger
           recovery with reason ``"ANOMALY_THRESHOLD_EXCEEDED"``.
        3. Both rules inspect history to select the earliest affected round
           and collect persistently suspicious client IDs.
        """
        # Advance round counter before processing
        self.current_round += 1
        current_round_num = self.current_round

        # Record this round in history BEFORE evaluation so that the current
        # round is included in look-back analysis.
        self._record_round(current_round_num, anomaly_scores)

        # --- Rule 1: Validation Override ---
        if self._loss_spike_priority and loss_spiked:
            return self._build_triggered_decision(
                trigger_reason="VALIDATION_LOSS_SPIKE",
                val_metrics=val_metrics,
            )

        # --- Rule 2: Anomaly-Based Trigger ---
        suspicious_this_round = self._get_suspicious_clients(anomaly_scores)
        if suspicious_this_round:
            return self._build_triggered_decision(
                trigger_reason="ANOMALY_THRESHOLD_EXCEEDED",
                val_metrics=val_metrics,
            )

        # --- No trigger ---
        self.recovery_state = RecoveryState.NOT_REQUIRED
        return RecoveryDecision(
            state=RecoveryState.NOT_REQUIRED,
            recovery_round=None,
            quarantined_clients=[],
            trigger_reason="NOT_REQUIRED",
            validation_metrics=val_metrics,
            evaluated_at=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # Properties (read-only views of internal state)
    # ------------------------------------------------------------------

    @property
    def round_history(self) -> List[_RoundRecord]:
        """
        Return a list snapshot of the current round history (oldest first).

        The list is a shallow copy; modifying it does not affect internal
        state.
        """
        return list(self._round_history)

    @property
    def anomaly_threshold(self) -> float:
        """Active anomaly threshold used for suspicious-client classification."""
        return self._anomaly_threshold

    @property
    def lookback_rounds(self) -> int:
        """Number of recent rounds inspected during historical analysis."""
        return self._lookback_rounds

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _record_round(
        self,
        round_num: int,
        scores: Dict[str, float],
    ) -> None:
        """
        Append the current round's anomaly scores to the rolling history.

        Parameters
        ----------
        round_num : int
            The round identifier to store.
        scores : Dict[str, float]
            Anomaly scores for every client in this round.
        """
        self._round_history.append(
            {
                "round": round_num,
                "scores": dict(scores),  # defensive copy
            }
        )

    def _get_suspicious_clients(
        self,
        scores: Dict[str, float],
    ) -> List[str]:
        """
        Return client IDs whose anomaly score meets or exceeds the threshold.

        Parameters
        ----------
        scores : Dict[str, float]
            Anomaly scores for a single round.

        Returns
        -------
        List[str]
            Sorted list of suspicious client IDs.  Empty if none exceed the
            threshold.
        """
        return sorted(
            client_id
            for client_id, score in scores.items()
            if score >= self._anomaly_threshold
        )

    def _collect_quarantined_clients(self) -> List[str]:
        """
        Inspect the look-back window and identify clients that appear
        suspicious in at least one recent round.

        Rule 3 (Historical Client Analysis): A client is quarantined if its
        anomaly score exceeded the threshold in any round within the
        look-back window.

        Returns
        -------
        List[str]
            Deduplicated, sorted list of client IDs to quarantine.
        """
        quarantined: set = set()
        for record in self._round_history:
            quarantined.update(
                self._get_suspicious_clients(record["scores"])
            )
        return sorted(quarantined)

    def _find_recovery_round(self, quarantined_clients: List[str]) -> Optional[int]:
        """
        Identify the earliest recent round that contained at least one of
        the quarantined clients.

        Step 4 (Target Recovery Round Selection): Walk the history from
        oldest to newest and return the first round number where any
        quarantined client has a score >= threshold.

        Parameters
        ----------
        quarantined_clients : List[str]
            Client IDs selected for quarantine.

        Returns
        -------
        Optional[int]
            The round number to recover to, or ``None`` if the history is
            empty or no matching round is found.
        """
        if not quarantined_clients:
            return None

        quarantined_set = set(quarantined_clients)

        # History is ordered oldest-first (deque insertion order).
        for record in self._round_history:
            suspicious_in_round = self._get_suspicious_clients(record["scores"])
            if quarantined_set.intersection(suspicious_in_round):
                return record["round"]

        # Fallback: no matching round found in history (edge case)
        return None

    def _build_triggered_decision(
        self,
        trigger_reason: str,
        val_metrics: Optional[Dict[str, float]],
    ) -> RecoveryDecision:
        """
        Construct a TRIGGERED ``RecoveryDecision``.

        Shared by both Rule 1 (validation spike) and Rule 2 (anomaly
        threshold exceeded).  Performs historical client analysis (Rule 3)
        and recovery round selection (Step 4) in both cases.

        Parameters
        ----------
        trigger_reason : str
            Reason code string to embed in the decision.
        val_metrics : Optional[Dict[str, float]]
            Validation metrics pass-through.

        Returns
        -------
        RecoveryDecision
            Fully populated TRIGGERED decision.
        """
        self.recovery_state = RecoveryState.TRIGGERED

        quarantined_clients = self._collect_quarantined_clients()
        recovery_round = self._find_recovery_round(quarantined_clients)

        return RecoveryDecision(
            state=RecoveryState.TRIGGERED,
            recovery_round=recovery_round,
            quarantined_clients=quarantined_clients,
            trigger_reason=trigger_reason,
            validation_metrics=val_metrics,
            evaluated_at=datetime.now(timezone.utc),
        )
