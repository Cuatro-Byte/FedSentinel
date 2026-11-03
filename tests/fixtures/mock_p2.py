"""
Minimal deterministic P2 (Attack) test stub.

This is a TEST FIXTURE ONLY — not a real attack implementation.
It exists solely to exercise Person 4's orchestration, persistence, and API.

It will be replaced by Person 2's real module during integration.
"""

from backend.adapters.interfaces import AttackInterface
from core.models import ModelUpdate


class MockP2Attack(AttackInterface):
    """Minimal deterministic Attack stub for testing P4 infrastructure.

    Marks designated attacker clients by scaling up their parameter values.
    This is NOT a real attack algorithm.
    """

    def __init__(self):
        self._attacker_client_ids: list[str] = []

    def apply_attacks(self, updates: list[ModelUpdate],
                      round_id: int,
                      config: dict) -> list[ModelUpdate]:
        """Mark attacker updates with abnormally large parameters.

        Only activates at or after the configured start_round.
        """
        if not config.get("enabled", True):
            return updates

        start_round = config.get("start_round", 5)
        if round_id < start_round:
            return updates

        attacker_ids = self.get_attacker_ids(round_id, config)
        self._attacker_client_ids = attacker_ids
        intensity = config.get("intensity", 0.8)

        modified_updates = []
        for update in updates:
            if update.client_id in attacker_ids:
                # Scale parameters to simulate attack — NOT real attack math
                attacked_params = {}
                for key, values in update.parameters.items():
                    if isinstance(values, list):
                        attacked_params[key] = [v * (10.0 * intensity) for v in values]
                    else:
                        attacked_params[key] = values * (10.0 * intensity)
                attacked_update = update.model_copy(
                    update={"parameters": attacked_params}
                )
                modified_updates.append(attacked_update)
            else:
                modified_updates.append(update)
        return modified_updates

    def get_attacker_ids(self, round_id: int, config: dict) -> list[str]:
        """Return deterministic attacker IDs.

        Ground truth for EVALUATION ONLY.
        """
        attacker_count = config.get("attacker_count", 1)
        client_count = config.get("client_count", 20)
        # Deterministic: last N clients are attackers
        return [f"client-{client_count - 1 - i}" for i in range(attacker_count)]
