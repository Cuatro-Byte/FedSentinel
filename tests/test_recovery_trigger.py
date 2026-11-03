"""
tests/test_impact_estimator.py
===============================
FedSentinel — Person B (Person 3)

Unit tests for ``RecoveryTriggerEngine``.

Covers:
  - no anomalies (all scores below threshold)
  - anomaly below threshold (boundary)
  - anomaly above threshold (single client)
  - multiple malicious clients
  - historical round tracking (look-back window)
  - custom threshold / lookback configuration
  - round counter increments correctly
  - round history max-length capping

All tests use only synthetic in-memory data.
No external project modules are imported.

Run with:
    python -m pytest tests/test_impact_estimator.py -v
"""

import sys
import os

# ---------------------------------------------------------------------------
# Path setup: allow running from the repository root without installing the
# package.  Works when tests/ and core/ are siblings under the same parent.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from core.sentinel.recovery_trigger import RecoveryTriggerEngine
from core.sentinel.recovery_models import RecoveryDecision, RecoveryState


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture()
def estimator() -> RecoveryTriggerEngine:
    """Return a fresh RecoveryTriggerEngine with default configuration."""
    return RecoveryTriggerEngine()


@pytest.fixture()
def strict_estimator() -> RecoveryTriggerEngine:
    """Return an RecoveryTriggerEngine with a very high threshold (0.99)."""
    return RecoveryTriggerEngine(anomaly_threshold=0.99, lookback_rounds=3)


@pytest.fixture()
def relaxed_estimator() -> RecoveryTriggerEngine:
    """Return an RecoveryTriggerEngine with a low threshold (0.50)."""
    return RecoveryTriggerEngine(anomaly_threshold=0.50, lookback_rounds=2)


# ===========================================================================
# TC-01 — No anomalies: all scores well below threshold
# ===========================================================================


class TestNoAnomalies:
    """All client scores are far below the configured threshold."""

    def test_state_not_required(self, estimator: RecoveryTriggerEngine) -> None:
        scores = {"client_01": 0.10, "client_02": 0.20, "client_03": 0.05}
        decision = estimator.evaluate_recovery_need(anomaly_scores=scores)
        assert decision.state is RecoveryState.NOT_REQUIRED

    def test_quarantined_clients_empty(self, estimator: RecoveryTriggerEngine) -> None:
        scores = {"client_01": 0.10, "client_02": 0.20}
        decision = estimator.evaluate_recovery_need(anomaly_scores=scores)
        assert decision.quarantined_clients == []

    def test_recovery_round_none(self, estimator: RecoveryTriggerEngine) -> None:
        scores = {"client_01": 0.10, "client_02": 0.20}
        decision = estimator.evaluate_recovery_need(anomaly_scores=scores)
        assert decision.recovery_round is None

    def test_trigger_reason_not_required(self, estimator: RecoveryTriggerEngine) -> None:
        scores = {"client_01": 0.10, "client_02": 0.20}
        decision = estimator.evaluate_recovery_need(anomaly_scores=scores)
        assert decision.trigger_reason == "NOT_REQUIRED"

    def test_is_triggered_false(self, estimator: RecoveryTriggerEngine) -> None:
        scores = {"client_01": 0.10}
        decision = estimator.evaluate_recovery_need(anomaly_scores=scores)
        assert decision.is_triggered is False

    def test_validation_metrics_passed_through(self, estimator: RecoveryTriggerEngine) -> None:
        scores = {"client_01": 0.10}
        metrics = {"val_loss": 0.45, "val_acc": 0.88, "loss_delta": 0.01, "acc_delta": 0.002}
        decision = estimator.evaluate_recovery_need(
            anomaly_scores=scores,
            val_metrics=metrics,
            loss_spiked=False,
        )
        assert decision.validation_metrics == metrics

    def test_estimator_state_updated(self, estimator: RecoveryTriggerEngine) -> None:
        scores = {"client_01": 0.10}
        estimator.evaluate_recovery_need(anomaly_scores=scores)
        assert estimator.recovery_state is RecoveryState.NOT_REQUIRED


# ===========================================================================
# TC-02 — Anomaly score exactly below threshold (boundary)
# ===========================================================================


class TestBelowThreshold:
    """Client score is strictly below the threshold — NOT at or above it."""

    THRESHOLD = 0.80  # default

    def test_score_just_below_threshold_not_triggered(self, estimator: RecoveryTriggerEngine) -> None:
        # 0.799 < 0.80 → must not trigger
        scores = {"client_07": 0.799}
        decision = estimator.evaluate_recovery_need(anomaly_scores=scores)
        assert decision.state is RecoveryState.NOT_REQUIRED

    def test_score_zero_not_triggered(self, estimator: RecoveryTriggerEngine) -> None:
        scores = {"client_01": 0.0, "client_02": 0.0}
        decision = estimator.evaluate_recovery_need(anomaly_scores=scores)
        assert decision.state is RecoveryState.NOT_REQUIRED

    def test_score_mixed_all_below_threshold(self, estimator: RecoveryTriggerEngine) -> None:
        scores = {
            "client_01": 0.14,
            "client_02": 0.18,
            "client_03": 0.55,
            "client_04": 0.79,
        }
        decision = estimator.evaluate_recovery_need(anomaly_scores=scores)
        assert decision.state is RecoveryState.NOT_REQUIRED
        assert decision.quarantined_clients == []


# ===========================================================================
# TC-03 — Single anomaly at or above threshold
# ===========================================================================


class TestAboveThreshold:
    """A single client score meets or exceeds the threshold."""

    def test_score_exactly_at_threshold_triggers(self, estimator: RecoveryTriggerEngine) -> None:
        # 0.80 >= 0.80 → must trigger
        scores = {"client_07": 0.80}
        decision = estimator.evaluate_recovery_need(anomaly_scores=scores)
        assert decision.state is RecoveryState.TRIGGERED

    def test_score_above_threshold_triggers(self, estimator: RecoveryTriggerEngine) -> None:
        scores = {"client_07": 0.95}
        decision = estimator.evaluate_recovery_need(anomaly_scores=scores)
        assert decision.state is RecoveryState.TRIGGERED

    def test_client_id_present_in_quarantine(self, estimator: RecoveryTriggerEngine) -> None:
        scores = {"client_07": 0.95}
        decision = estimator.evaluate_recovery_need(anomaly_scores=scores)
        assert "client_07" in decision.quarantined_clients

    def test_reason_anomaly_threshold(self, estimator: RecoveryTriggerEngine) -> None:
        scores = {"client_07": 0.95}
        decision = estimator.evaluate_recovery_need(anomaly_scores=scores)
        assert decision.trigger_reason == "ANOMALY_THRESHOLD_EXCEEDED"

    def test_is_triggered_true(self, estimator: RecoveryTriggerEngine) -> None:
        scores = {"client_07": 0.95}
        decision = estimator.evaluate_recovery_need(anomaly_scores=scores)
        assert decision.is_triggered is True

    def test_recovery_round_is_current(self, estimator: RecoveryTriggerEngine) -> None:
        scores = {"client_07": 0.95}
        decision = estimator.evaluate_recovery_need(anomaly_scores=scores)
        # First call → round 1 → recovery_round should be 1
        assert decision.recovery_round == 1

    def test_estimator_state_triggered(self, estimator: RecoveryTriggerEngine) -> None:
        scores = {"client_07": 0.95}
        estimator.evaluate_recovery_need(anomaly_scores=scores)
        assert estimator.recovery_state is RecoveryState.TRIGGERED


# ===========================================================================
# TC-04 — Multiple malicious clients
# ===========================================================================


class TestMultipleMaliciousClients:
    """Several clients simultaneously exceed the threshold."""

    def test_all_malicious_clients_quarantined(self, estimator: RecoveryTriggerEngine) -> None:
        scores = {
            "client_01": 0.82,
            "client_02": 0.14,
            "client_03": 0.91,
            "client_04": 0.88,
        }
        decision = estimator.evaluate_recovery_need(anomaly_scores=scores)
        assert decision.state is RecoveryState.TRIGGERED
        assert set(decision.quarantined_clients) == {"client_01", "client_03", "client_04"}

    def test_clean_clients_not_quarantined(self, estimator: RecoveryTriggerEngine) -> None:
        scores = {
            "client_01": 0.82,
            "client_02": 0.14,  # below threshold
        }
        decision = estimator.evaluate_recovery_need(anomaly_scores=scores)
        assert "client_02" not in decision.quarantined_clients

    def test_quarantined_list_is_sorted(self, estimator: RecoveryTriggerEngine) -> None:
        scores = {"client_05": 0.91, "client_01": 0.85, "client_03": 0.80}
        decision = estimator.evaluate_recovery_need(anomaly_scores=scores)
        assert decision.quarantined_clients == sorted(decision.quarantined_clients)

    def test_validation_metrics_not_modified(self, estimator: RecoveryTriggerEngine) -> None:
        scores = {"client_01": 0.91}
        original_metrics = {
            "val_loss": 0.91,
            "val_acc": 0.62,
            "loss_delta": 0.19,
            "acc_delta": -0.08,
        }
        metrics_copy = dict(original_metrics)
        decision = estimator.evaluate_recovery_need(
            anomaly_scores=scores, val_metrics=metrics_copy
        )
        assert decision.validation_metrics == original_metrics


# ===========================================================================
# TC-05 — Historical round tracking
# ===========================================================================


class TestHistoricalRoundTracking:
    """Verify that the rolling history is populated and inspected correctly."""

    def test_history_grows_with_rounds(self, estimator: RecoveryTriggerEngine) -> None:
        for i in range(1, 4):
            estimator.evaluate_recovery_need(
                anomaly_scores={f"client_{i:02d}": 0.10}
            )
        assert len(estimator.round_history) == 3

    def test_history_capped_at_lookback(self, estimator: RecoveryTriggerEngine) -> None:
        # Default lookback = 3; push 5 rounds
        for i in range(5):
            estimator.evaluate_recovery_need(
                anomaly_scores={f"client_{i:02d}": 0.10}
            )
        assert len(estimator.round_history) == 3

    def test_history_contains_correct_rounds(self, estimator: RecoveryTriggerEngine) -> None:
        for i in range(1, 4):
            estimator.evaluate_recovery_need(
                anomaly_scores={f"client_{i:02d}": 0.10}
            )
        round_nums = [r["round"] for r in estimator.round_history]
        assert round_nums == [1, 2, 3]

    def test_history_scores_stored_correctly(self, estimator: RecoveryTriggerEngine) -> None:
        scores = {"client_07": 0.95, "client_02": 0.10}
        estimator.evaluate_recovery_need(anomaly_scores=scores)
        stored = estimator.round_history[0]["scores"]
        assert stored == scores

    def test_quarantined_from_history_across_rounds(self, estimator: RecoveryTriggerEngine) -> None:
        """
        Round 1: client_07 is suspicious.
        Round 2: client_07 is fine, but client_03 spikes.
        Round 3: client_03 remains suspicious.
        Expected quarantine: both client_07 (from round 1 history when round 3 triggers)
        AND client_03 (current + round 2 history).
        """
        # Round 1 — suspicious client_07
        estimator.evaluate_recovery_need(
            anomaly_scores={"client_07": 0.85, "client_03": 0.20}
        )
        # Round 2 — client_07 drops below threshold, client_03 spikes
        estimator.evaluate_recovery_need(
            anomaly_scores={"client_07": 0.30, "client_03": 0.90}
        )
        # Round 3 — client_03 still above threshold → triggers
        decision = estimator.evaluate_recovery_need(
            anomaly_scores={"client_07": 0.20, "client_03": 0.88}
        )
        assert decision.state is RecoveryState.TRIGGERED
        # client_03 appeared in rounds 2 and 3 → quarantined
        assert "client_03" in decision.quarantined_clients
        # client_07 appeared in round 1 (within lookback=3) → quarantined
        assert "client_07" in decision.quarantined_clients

    def test_recovery_round_earliest_in_window(self) -> None:
        """
        Recovery round should be the EARLIEST round in the look-back window
        that contained a now-quarantined client.
        """
        est = RecoveryTriggerEngine(lookback_rounds=3)
        # Round 1: client_07 suspicious
        est.evaluate_recovery_need(anomaly_scores={"client_07": 0.85})
        # Round 2: clean
        est.evaluate_recovery_need(anomaly_scores={"client_01": 0.10})
        # Round 3: client_07 spikes again → triggers; earliest = round 1
        decision = est.evaluate_recovery_need(
            anomaly_scores={"client_07": 0.92}
        )
        assert decision.state is RecoveryState.TRIGGERED
        assert decision.recovery_round == 1

    def test_round_counter_increments(self, estimator: RecoveryTriggerEngine) -> None:
        assert estimator.current_round == 0
        estimator.evaluate_recovery_need(anomaly_scores={"c": 0.1})
        assert estimator.current_round == 1
        estimator.evaluate_recovery_need(anomaly_scores={"c": 0.1})
        assert estimator.current_round == 2


# ===========================================================================
# TC-06 — Custom threshold and lookback configuration
# ===========================================================================


class TestCustomConfiguration:

    def test_high_threshold_does_not_trigger_moderate_scores(
        self, strict_estimator: RecoveryTriggerEngine
    ) -> None:
        # threshold=0.99 → only scores >= 0.99 trigger
        scores = {"client_01": 0.95}
        decision = strict_estimator.evaluate_recovery_need(anomaly_scores=scores)
        assert decision.state is RecoveryState.NOT_REQUIRED

    def test_high_threshold_triggers_on_extreme_score(
        self, strict_estimator: RecoveryTriggerEngine
    ) -> None:
        scores = {"client_01": 0.99}
        decision = strict_estimator.evaluate_recovery_need(anomaly_scores=scores)
        assert decision.state is RecoveryState.TRIGGERED

    def test_low_threshold_triggers_on_moderate_scores(
        self, relaxed_estimator: RecoveryTriggerEngine
    ) -> None:
        # threshold=0.50 → scores >= 0.50 trigger
        scores = {"client_01": 0.55}
        decision = relaxed_estimator.evaluate_recovery_need(anomaly_scores=scores)
        assert decision.state is RecoveryState.TRIGGERED

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            RecoveryTriggerEngine(anomaly_threshold=0.0)
        with pytest.raises(ValueError):
            RecoveryTriggerEngine(anomaly_threshold=1.1)

    def test_invalid_lookback_raises(self) -> None:
        with pytest.raises(ValueError):
            RecoveryTriggerEngine(lookback_rounds=0)

    def test_lookback_2_caps_history(self, relaxed_estimator: RecoveryTriggerEngine) -> None:
        for i in range(5):
            relaxed_estimator.evaluate_recovery_need(
                anomaly_scores={f"client_{i:02d}": 0.10}
            )
        assert len(relaxed_estimator.round_history) == 2


# ===========================================================================
# TC-07 — Empty anomaly scores
# ===========================================================================


class TestEmptyScores:

    def test_empty_scores_not_triggered(self, estimator: RecoveryTriggerEngine) -> None:
        decision = estimator.evaluate_recovery_need(anomaly_scores={})
        assert decision.state is RecoveryState.NOT_REQUIRED

    def test_empty_scores_no_quarantined_clients(self, estimator: RecoveryTriggerEngine) -> None:
        decision = estimator.evaluate_recovery_need(anomaly_scores={})
        assert decision.quarantined_clients == []

    def test_empty_scores_recovery_round_none(self, estimator: RecoveryTriggerEngine) -> None:
        decision = estimator.evaluate_recovery_need(anomaly_scores={})
        assert decision.recovery_round is None


# ===========================================================================
# TC-08 — Optional val_metrics = None
# ===========================================================================


class TestOptionalMetrics:

    def test_metrics_none_does_not_raise(self, estimator: RecoveryTriggerEngine) -> None:
        decision = estimator.evaluate_recovery_need(
            anomaly_scores={"client_01": 0.10},
            val_metrics=None,
            loss_spiked=False,
        )
        assert decision.validation_metrics is None

    def test_metrics_none_on_trigger(self, estimator: RecoveryTriggerEngine) -> None:
        decision = estimator.evaluate_recovery_need(
            anomaly_scores={"client_01": 0.90},
            val_metrics=None,
        )
        assert decision.state is RecoveryState.TRIGGERED
        assert decision.validation_metrics is None


# ===========================================================================
# TC-09 — RecoveryDecision helpers
# ===========================================================================


class TestRecoveryDecisionHelpers:

    def test_summary_not_required(self, estimator: RecoveryTriggerEngine) -> None:
        decision = estimator.evaluate_recovery_need(anomaly_scores={"c": 0.1})
        summary = decision.summary()
        assert "NOT_REQUIRED" in summary

    def test_summary_triggered(self, estimator: RecoveryTriggerEngine) -> None:
        decision = estimator.evaluate_recovery_need(anomaly_scores={"c": 0.99})
        summary = decision.summary()
        assert "TRIGGERED" in summary

    def test_repr_contains_state(self, estimator: RecoveryTriggerEngine) -> None:
        decision = estimator.evaluate_recovery_need(anomaly_scores={"c": 0.1})
        assert "NOT_REQUIRED" in repr(decision)

    def test_evaluated_at_is_datetime(self, estimator: RecoveryTriggerEngine) -> None:
        from datetime import datetime as dt

        decision = estimator.evaluate_recovery_need(anomaly_scores={"c": 0.1})
        assert isinstance(decision.evaluated_at, dt)
