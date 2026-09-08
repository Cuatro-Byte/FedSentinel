"""
Abstract interfaces for P1, P2, P3 team modules.

These define the contracts that the orchestrator uses to coordinate
with each team's implementation. Real modules and test fixtures
both implement these interfaces.

Person 4 calls these interfaces. Person 4 does NOT implement
the algorithms behind them.
"""

from abc import ABC, abstractmethod
from typing import Any

from core.models import (
    ModelUpdate, DetectionResult, ImpactResult, RecoveryResult,
)
from core.models.enums import ResponseAction


class FLCoreInterface(ABC):
    """Interface for Person 1's FL Core module.

    Covers: training, aggregation, evaluation, recovery re-aggregation.
    Person 1 OWNS the implementation of these methods.
    """

    @abstractmethod
    def initialize_global_model(self, config: dict) -> str:
        """Initialize the global model. Returns initial model_version."""
        ...

    @abstractmethod
    def train_clients(self, run_id: str, round_id: int,
                      global_model_version: str,
                      client_ids: list[str],
                      config: dict) -> list[ModelUpdate]:
        """Simulate local training for all clients.
        Returns a list of ModelUpdate objects.
        """
        ...

    @abstractmethod
    def aggregate(self, updates: list[ModelUpdate],
                  detections: list[DetectionResult],
                  current_model_version: str) -> str:
        """Perform trust-aware aggregation.

        Args:
            updates: All model updates for this round.
            detections: List of DetectionResult objects from P3.
                        P1 uses the action field to weight/exclude updates.
            current_model_version: Current global model version.

        Returns:
            New model version string after aggregation.

        Note: P4 passes the trust/action information TO P1.
              P1 decides how to aggregate. P4 does NOT filter.
        """
        ...

    @abstractmethod
    def evaluate(self, model_version: str) -> dict:
        """Evaluate the global model.

        Returns dict with at least:
            - accuracy: float
            - loss: float
        """
        ...

    @abstractmethod
    def re_aggregate(self, updates: list[ModelUpdate],
                     excluded_client_ids: list[str],
                     previous_model_version: str) -> str:
        """Selective re-aggregation for recovery (owned by P1).

        Args:
            updates: Original round updates.
            excluded_client_ids: Clients to exclude (determined by P3).
            previous_model_version: The model version before recovery.

        Returns:
            Recovered model version string.
        """
        ...


class AttackInterface(ABC):
    """Interface for Person 2's Attack module.

    Covers: attack injection, attacker selection, poisoning.
    Person 2 OWNS the implementation of these methods.
    """

    @abstractmethod
    def apply_attacks(self, updates: list[ModelUpdate],
                      round_id: int,
                      config: dict) -> list[ModelUpdate]:
        """Apply attacks to designated client updates.

        Args:
            updates: Original model updates from P1 training.
            round_id: Current round number.
            config: Attack configuration.

        Returns:
            Modified list of ModelUpdate objects (some may be attacked).

        Note: Attack ground truth must NOT be passed to the detector
              during inference (Contract §5 Person 2 Critical Rule).
        """
        ...

    @abstractmethod
    def get_attacker_ids(self, round_id: int, config: dict) -> list[str]:
        """Return the client IDs designated as attackers for this round.

        This is ground truth for EVALUATION ONLY — not for detection.
        """
        ...

    def apply_data_attack(self, client_id: str, dataset: Any,
                          round_id: int, config: dict,
                          client_ids: list[str] | None = None) -> Any:
        """Apply data-level attack to a client's dataset before training.

        Returns the modified dataset if the client is an attacker with a
        data-level attack, or the original dataset otherwise.
        """
        return dataset


class SentinelInterface(ABC):
    """Interface for Person 3's Sentinel module.

    Covers: detection, impact estimation, response decision, recovery decision.
    Person 3 OWNS the implementation of these methods.
    Person 4 does NOT implement the algorithms.
    """

    @abstractmethod
    def detect(self, updates: list[ModelUpdate],
               round_id: int) -> list[DetectionResult]:
        """Run detection pipeline on model updates.

        Returns a DetectionResult for each update.
        """
        ...

    @abstractmethod
    def estimate_impact(self, updates: list[ModelUpdate],
                        detections: list[DetectionResult],
                        round_id: int) -> list[ImpactResult]:
        """Estimate impact of each update.

        Impact is SEPARATE from threat (Contract §10 Important distinction).
        Returns an ImpactResult for each update.
        """
        ...

    @abstractmethod
    def decide_response(self, detections: list[DetectionResult],
                        impacts: list[ImpactResult]) -> dict[str, ResponseAction]:
        """Determine response action for each update.

        Returns a map of update_id → ResponseAction.
        The orchestrator passes this to P1 for trust-aware aggregation.
        """
        ...

    @abstractmethod
    def check_recovery(self, evaluation: dict,
                       detections: list[DetectionResult],
                       impacts: list[ImpactResult],
                       run_id: str, round_id: int,
                       model_version: str,
                       config: dict) -> RecoveryResult:
        """Determine if recovery is needed after aggregation.

        Person 3 owns the recovery trigger intelligence.
        Person 3 determines WHICH updates to exclude.

        Returns a RecoveryResult with recovery_status indicating
        whether recovery should proceed.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state across simulation runs."""
        ...
