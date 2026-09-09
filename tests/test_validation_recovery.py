"""
tests/test_validation_recovery.py
===================================
FedSentinel — Person B (Person 3)

Integration-style unit tests for the validation-aware recovery path of
``RecoveryTriggerEngine``.

Covers:
  - validation loss spike triggers recovery (Rule 1 override)
  - validation metrics preserved verbatim inside RecoveryDecision
  - recovery round identified correctly after a spike
  - empty history when spike occurs on round 1
  - optional metrics = None backward compatibility
  - LOSS_SPIKE_PRIORITY=False disables validation override
  - Sleeper Scenario simulation (Step 8):
        Rounds 1–3: low anomaly + loss_spiked=False → NOT_REQUIRED
        Round 4:    high anomaly + loss_spiked=True  → TRIGGERED,
                    quarantined client returned, recovery_round = 4

All tests use only synthetic in-memory data.
No external project modules are imported.

Run with:
    python -m pytest tests/test_validation_recovery.py -v
"""

import sys
import os

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from core.sentinel.recovery_trigger import RecoveryTriggerEngine
from core.sentinel.recovery_models import RecoveryDecision, RecoveryState


# ===========================================================================
# Helpers / shared fixtures
# ===========================================================================

BENIGN_SCORES = {
    "client_01": 0.14,
    "client_02": 0.18,
    "client_03": 0.22,
    "client_04": 0.09,
}

SPIKE_METRICS = {
    "val_loss": 0.91,
    "val_acc": 0.62,
    "loss_delta": 0.19,
    "acc_delta": -0.08,
}

NORMAL_METRICS = {
    "val_loss": 0.45,
    "val_acc": 0.88,
    "loss_delta": 0.01,
    "acc_delta": 0.002,
}


@pytest.fixture()
def estimator() -> RecoveryTriggerEngine:
    """Return a fresh RecoveryTriggerEngine with default configuration."""
    return RecoveryTriggerEngine()


# ===========================================================================
# TC-V01 — Validation loss spike triggers recovery (Rule 1)
# ===========================================================================


class TestValidationLossSpike:
    """loss_spiked=True must immediately trigger recovery."""

    def test_spike_triggers_recovery(self, estimator: RecoveryTriggerEngine) -> None:
        decision = estimator.evaluate_recovery_need(
            anomaly_scores=BENIGN_SCORES,
            val_metrics=SPIKE_METRICS,
            loss_spiked=True,
        )
        assert decision.state is RecoveryState.TRIGGERED

    def test_spike_reason_validation_loss_spike(self, estimator: RecoveryTriggerEngine) -> None:
        decision = estimator.evaluate_recovery_need(
            anomaly_scores=BENIGN_SCORES,
            val_metrics=SPIKE_METRICS,
            loss_spiked=True,
        )
        assert decision.trigger_reason == "VALIDATION_LOSS_SPIKE"

    def test_spike_overrides_low_anomaly_scores(self, estimator: RecoveryTriggerEngine) -> None:
        """
        Even with all anomaly scores well below threshold, a loss spike must
        trigger recovery.
        """
        safe_scores = {"client_01": 0.05, "client_02": 0.10}
        decision = estimator.evaluate_recovery_need(
            anomaly_scores=safe_scores,
            val_metrics=SPIKE_METRICS,
            loss_spiked=True,
        )
        assert decision.state is RecoveryState.TRIGGERED
        assert decision.trigger_reason == "VALIDATION_LOSS_SPIKE"

    def test_spike_is_triggered_property(self, estimator: RecoveryTriggerEngine) -> None:
        decision = estimator.evaluate_recovery_need(
            anomaly_scores=BENIGN_SCORES,
            loss_spiked=True,
        )
        assert decision.is_triggered is True

    def test_no_spike_and_benign_not_triggered(self, estimator: RecoveryTriggerEngine) -> None:
        decision = estimator.evaluate_recovery_need(
            anomaly_scores=BENIGN_SCORES,
            val_metrics=NORMAL_METRICS,
            loss_spiked=False,
        )
        assert decision.state is RecoveryState.NOT_REQUIRED


# ===========================================================================
# TC-V02 — Validation metrics preserved verbatim
# ===========================================================================


class TestValidationMetricsPreserved:
    """Metrics must be stored as-is inside RecoveryDecision."""

    def test_metrics_exact_match_on_spike(self, estimator: RecoveryTriggerEngine) -> None:
        decision = estimator.evaluate_recovery_need(
            anomaly_scores=BENIGN_SCORES,
            val_metrics=SPIKE_METRICS,
            loss_spiked=True,
        )
        assert decision.validation_metrics == SPIKE_METRICS

    def test_metrics_exact_match_no_trigger(self, estimator: RecoveryTriggerEngine) -> None:
        decision = estimator.evaluate_recovery_need(
            anomaly_scores=BENIGN_SCORES,
            val_metrics=NORMAL_METRICS,
            loss_spiked=False,
        )
        assert decision.validation_metrics == NORMAL_METRICS

    def test_metrics_val_loss_preserved(self, estimator: RecoveryTriggerEngine) -> None:
        decision = estimator.evaluate_recovery_need(
            anomaly_scores=BENIGN_SCORES,
            val_metrics=SPIKE_METRICS,
            loss_spiked=True,
        )
        assert decision.validation_metrics["val_loss"] == SPIKE_METRICS["val_loss"]

    def test_metrics_val_acc_preserved(self, estimator: RecoveryTriggerEngine) -> None:
        decision = estimator.evaluate_recovery_need(
            anomaly_scores=BENIGN_SCORES,
            val_metrics=SPIKE_METRICS,
            loss_spiked=True,
        )
        assert decision.validation_metrics["val_acc"] == SPIKE_METRICS["val_acc"]

    def test_metrics_loss_delta_preserved(self, estimator: RecoveryTriggerEngine) -> None:
        decision = estimator.evaluate_recovery_need(
            anomaly_scores=BENIGN_SCORES,
            val_metrics=SPIKE_METRICS,
            loss_spiked=True,
        )
        assert decision.validation_metrics["loss_delta"] == SPIKE_METRICS["loss_delta"]

    def test_metrics_acc_delta_preserved(self, estimator: RecoveryTriggerEngine) -> None:
        decision = estimator.evaluate_recovery_need(
            anomaly_scores=BENIGN_SCORES,
            val_metrics=SPIKE_METRICS,
            loss_spiked=True,
        )
        assert decision.validation_metrics["acc_delta"] == SPIKE_METRICS["acc_delta"]

    def test_metrics_not_mutated_by_estimator(self, estimator: RecoveryTriggerEngine) -> None:
        """The estimator must not alter the caller's dictionary."""
        original = dict(SPIKE_METRICS)
        estimator.evaluate_recovery_need(
            anomaly_scores=BENIGN_SCORES,
            val_metrics=original,
            loss_spiked=True,
        )
        assert original == SPIKE_METRICS


# ===========================================================================
# TC-V03 — Recovery round identified after a spike
# ===========================================================================


class TestRecoveryRoundIdentification:
    """
    Verify recovery round selection when a loss spike fires after several
    previous rounds.
    """

    def test_spike_on_round_1_with_suspicious_client_returns_round_1(
        self, estimator: RecoveryTriggerEngine
    ) -> None:
        """
        On round 1 a suspicious client (above threshold) is present AND a
        loss spike fires.  The current round is stored into history before
        evaluation, so recovery_round must be 1.
        """
        scores_with_suspect = {"client_07": 0.92, "client_01": 0.10}
        decision = estimator.evaluate_recovery_need(
            anomaly_scores=scores_with_suspect,
            val_metrics=SPIKE_METRICS,
            loss_spiked=True,
        )
        # client_07 is above threshold → quarantined; earliest round = 1
        assert decision.recovery_round == 1
        assert "client_07" in decision.quarantined_clients

    def test_spike_after_benign_rounds_returns_earliest_suspicious(self) -> None:
        """
        Round 1: benign
        Round 2: client_07 is suspicious (above threshold)
        Round 3: benign
        Round 4: loss_spiked=True → recovery_round should be 2
                 (earliest round with a now-quarantined client)
        """
        est = RecoveryTriggerEngine(lookback_rounds=4)

        # Round 1 — benign
        est.evaluate_recovery_need(
            anomaly_scores={"client_01": 0.10, "client_07": 0.20}
        )
        # Round 2 — client_07 above threshold (stored in history)
        est.evaluate_recovery_need(
            anomaly_scores={"client_01": 0.10, "client_07": 0.85}
        )
        # Round 3 — benign again
        est.evaluate_recovery_need(
            anomaly_scores={"client_01": 0.10, "client_07": 0.20}
        )
        # Round 4 — loss spike
        decision = est.evaluate_recovery_need(
            anomaly_scores={"client_01": 0.10, "client_07": 0.20},
            val_metrics=SPIKE_METRICS,
            loss_spiked=True,
        )
        assert decision.state is RecoveryState.TRIGGERED
        # client_07 appeared above threshold in round 2 → earliest = 2
        assert "client_07" in decision.quarantined_clients
        assert decision.recovery_round == 2

    def test_spike_with_no_suspicious_history_returns_current_round_or_none(
        self, estimator: RecoveryTriggerEngine
    ) -> None:
        """
        If all anomaly scores are always below threshold but a spike fires,
        there are no quarantined clients → recovery_round should be None.
        """
        decision = estimator.evaluate_recovery_need(
            anomaly_scores={"client_01": 0.05},
            val_metrics=SPIKE_METRICS,
            loss_spiked=True,
        )
        # No suspicious clients in history → recovery_round is None
        assert decision.recovery_round is None


# ===========================================================================
# TC-V04 — Empty history
# ===========================================================================


class TestEmptyHistory:
    """Edge cases when no prior rounds have been recorded."""

    def test_spike_on_first_round_recovery_round_is_1(
        self, estimator: RecoveryTriggerEngine
    ) -> None:
        """
        On round 1 with suspicious clients and a spike, recovery_round
        must be 1 (current round recorded into history before evaluation).
        """
        scores = {"client_07": 0.92}
        decision = estimator.evaluate_recovery_need(
            anomaly_scores=scores,
            val_metrics=SPIKE_METRICS,
            loss_spiked=True,
        )
        assert decision.recovery_round == 1

    def test_spike_with_all_clean_first_round_recovery_none(
        self, estimator: RecoveryTriggerEngine
    ) -> None:
        decision = estimator.evaluate_recovery_need(
            anomaly_scores={"client_01": 0.05},
            val_metrics=SPIKE_METRICS,
            loss_spiked=True,
        )
        assert decision.recovery_round is None

    def test_history_empty_before_any_call(self, estimator: RecoveryTriggerEngine) -> None:
        assert len(estimator.round_history) == 0


# ===========================================================================
# TC-V05 — Optional metrics = None (backward compatibility)
# ===========================================================================


class TestOptionalMetricsBackwardCompat:

    def test_no_metrics_no_spike_backward_compat(self, estimator: RecoveryTriggerEngine) -> None:
        """Calling with only anomaly_scores (old API) must not raise."""
        decision = estimator.evaluate_recovery_need(
            anomaly_scores={"client_01": 0.10}
        )
        assert decision.state is RecoveryState.NOT_REQUIRED
        assert decision.validation_metrics is None

    def test_no_metrics_with_spike_triggers(self, estimator: RecoveryTriggerEngine) -> None:
        """loss_spiked=True with val_metrics=None must still trigger."""
        decision = estimator.evaluate_recovery_need(
            anomaly_scores={"client_01": 0.10},
            loss_spiked=True,
        )
        assert decision.state is RecoveryState.TRIGGERED
        assert decision.validation_metrics is None

    def test_metrics_none_anomaly_trigger(self, estimator: RecoveryTriggerEngine) -> None:
        decision = estimator.evaluate_recovery_need(
            anomaly_scores={"client_01": 0.95},
            val_metrics=None,
            loss_spiked=False,
        )
        assert decision.state is RecoveryState.TRIGGERED
        assert decision.validation_metrics is None


# ===========================================================================
# TC-V06 — LOSS_SPIKE_PRIORITY = False disables Rule 1
# ===========================================================================


class TestLossSpikePriorityDisabled:

    def test_spike_not_triggered_when_priority_disabled(self) -> None:
        """
        When loss_spike_priority=False, a spike alone must NOT trigger
        recovery if anomaly scores are below threshold.
        """
        est = RecoveryTriggerEngine(loss_spike_priority=False)
        decision = est.evaluate_recovery_need(
            anomaly_scores=BENIGN_SCORES,
            val_metrics=SPIKE_METRICS,
            loss_spiked=True,
        )
        assert decision.state is RecoveryState.NOT_REQUIRED

    def test_anomaly_still_triggers_when_priority_disabled(self) -> None:
        est = RecoveryTriggerEngine(loss_spike_priority=False)
        scores = {"client_07": 0.92}
        decision = est.evaluate_recovery_need(
            anomaly_scores=scores,
            loss_spiked=True,
        )
        assert decision.state is RecoveryState.TRIGGERED
        assert decision.trigger_reason == "ANOMALY_THRESHOLD_EXCEEDED"


# ===========================================================================
# TC-V07 — Sleeper Scenario Simulation (Step 8)
# ===========================================================================


class TestSleeperScenarioSimulation:
    """
    Simulates a sleeper-agent attack scenario that mimics Person 2's
    Sleeper Scenario verification.

    Timeline:
        Rounds 1–3: low anomaly scores; loss_spiked = False → NOT_REQUIRED
        Round 4:    high anomaly score for attacker_client; loss_spiked = True → TRIGGERED

    Assertions:
        - rounds 1–3 produce NOT_REQUIRED decisions
        - round 4 produces TRIGGERED
        - attacker_client is in quarantined_clients
        - recovery_round is 4 (first round attacker was flagged)
    """

    ATTACKER = "client_07"

    SLEEPER_ROUNDS = [
        # (round, scores, val_metrics, loss_spiked)
        (
            1,
            {"client_01": 0.14, "client_02": 0.18, "client_07": 0.13},
            NORMAL_METRICS,
            False,
        ),
        (
            2,
            {"client_01": 0.12, "client_02": 0.20, "client_07": 0.15},
            NORMAL_METRICS,
            False,
        ),
        (
            3,
            {"client_01": 0.16, "client_02": 0.17, "client_07": 0.19},
            NORMAL_METRICS,
            False,
        ),
        (
            4,
            {"client_01": 0.10, "client_02": 0.14, "client_07": 0.95},
            SPIKE_METRICS,
            True,
        ),
    ]

    def _run_simulation(self) -> list:
        """
        Run the sleeper simulation and return a list of RecoveryDecision
        objects, one per round.
        """
        est = RecoveryTriggerEngine()
        decisions = []
        for _rnd, scores, metrics, spiked in self.SLEEPER_ROUNDS:
            decision = est.evaluate_recovery_need(
                anomaly_scores=scores,
                val_metrics=metrics,
                loss_spiked=spiked,
            )
            decisions.append(decision)
        return decisions

    def test_rounds_1_to_3_not_required(self) -> None:
        decisions = self._run_simulation()
        for i in range(3):
            assert decisions[i].state is RecoveryState.NOT_REQUIRED, (
                f"Round {i + 1} should be NOT_REQUIRED, "
                f"got {decisions[i].state}"
            )

    def test_round_4_triggered(self) -> None:
        decisions = self._run_simulation()
        assert decisions[3].state is RecoveryState.TRIGGERED

    def test_round_4_attacker_quarantined(self) -> None:
        decisions = self._run_simulation()
        assert self.ATTACKER in decisions[3].quarantined_clients

    def test_round_4_recovery_round_is_4(self) -> None:
        """
        The attacker first appears above threshold in round 4 (rounds 1–3 were
        below threshold), so the earliest affected round is 4.
        """
        decisions = self._run_simulation()
        assert decisions[3].recovery_round == 4

    def test_round_4_reason_is_validation_loss_spike(self) -> None:
        decisions = self._run_simulation()
        assert decisions[3].trigger_reason == "VALIDATION_LOSS_SPIKE"

    def test_round_4_metrics_preserved(self) -> None:
        decisions = self._run_simulation()
        assert decisions[3].validation_metrics == SPIKE_METRICS

    def test_clean_clients_not_quarantined_round_4(self) -> None:
        decisions = self._run_simulation()
        quarantined = decisions[3].quarantined_clients
        assert "client_01" not in quarantined
        assert "client_02" not in quarantined

    def test_is_triggered_property_round_4(self) -> None:
        decisions = self._run_simulation()
        assert decisions[3].is_triggered is True

    def test_is_triggered_property_rounds_1_to_3_false(self) -> None:
        decisions = self._run_simulation()
        for i in range(3):
            assert decisions[i].is_triggered is False
