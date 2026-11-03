"""
Minimal deterministic P1 (FL Core) test stub.

This is a TEST FIXTURE ONLY — not a real FL implementation.
It exists solely to exercise Person 4's orchestration, persistence, and API.

It will be replaced by Person 1's real module during integration.
"""

import uuid
from datetime import datetime

from backend.adapters.interfaces import FLCoreInterface
from core.models import ModelUpdate
from core.models.enums import ResponseAction


class MockP1FL(FLCoreInterface):
    """Minimal deterministic FL Core stub for testing P4 infrastructure."""

    def __init__(self):
        self._model_version_counter = 0
        self._base_accuracy = 0.85
        self._base_loss = 0.35

    def initialize_global_model(self, config: dict) -> str:
        self._model_version_counter = 0
        return "model-v0"

    def train_clients(self, run_id: str, round_id: int,
                      global_model_version: str,
                      client_ids: list[str],
                      config: dict) -> list[ModelUpdate]:
        updates = []
        for i, cid in enumerate(client_ids):
            update = ModelUpdate(
                update_id=f"{run_id}-r{round_id}-{cid}",
                run_id=run_id,
                round_id=round_id,
                client_id=cid,
                model_version=global_model_version,
                base_model_version=global_model_version,
                parameters={
                    "layer1.weight": [0.01 * (i + 1)] * 10,
                    "layer1.bias": [0.001 * (i + 1)] * 5,
                },
                sample_count=100 + i * 10,
                local_loss=0.3 + 0.01 * i,
                local_accuracy=0.87 - 0.01 * i,
                training_epochs=config.get("local_epochs", 2),
                learning_rate=0.01,
                created_at=datetime.utcnow(),
                metadata={"client_index": i},
            )
            updates.append(update)
        return updates

    def aggregate(self, updates: list[ModelUpdate],
                  actions: dict[str, ResponseAction],
                  current_model_version: str) -> str:
        """Deterministic aggregation stub.

        Simply increments model version. Does NOT implement real
        aggregation math — that is P1's responsibility.
        """
        self._model_version_counter += 1
        return f"model-v{self._model_version_counter}"

    def evaluate(self, model_version: str) -> dict:
        """Deterministic evaluation stub.

        Returns slightly varying accuracy/loss based on version number.
        """
        version_num = self._model_version_counter
        accuracy = min(self._base_accuracy + 0.01 * version_num, 0.98)
        loss = max(self._base_loss - 0.02 * version_num, 0.05)
        return {"accuracy": round(accuracy, 4), "loss": round(loss, 4)}

    def validate_candidate(self, model_version: str) -> tuple[dict, bool]:
        """Deterministic validation stub."""
        return {"val_loss": 0.40, "val_acc": 0.80, "loss_delta": 0.05, "acc_delta": 0.02}, False

    def re_aggregate(self, updates: list[ModelUpdate],
                     excluded_client_ids: list[str],
                     previous_model_version: str) -> str:
        """Deterministic re-aggregation stub for recovery testing."""
        self._model_version_counter += 1
        return f"{previous_model_version}-recovered-{self._model_version_counter}"
