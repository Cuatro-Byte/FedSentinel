"""
Scenario manager — configures and coordinates simulation scenarios.

Owned by Person 4 (Contract §5).
Translates configuration into the parameters used by the orchestrator.
Does NOT implement attack, detection, or FL algorithms.
"""

from backend.config import AppConfig


class ScenarioManager:
    """Manages simulation scenario configuration.

    Provides scenario-specific configs to the runner without
    implementing attack or detection logic.
    """

    def __init__(self, config: AppConfig):
        self.config = config

    def get_simulation_params(self) -> dict:
        """Return the simulation parameters."""
        return {
            "client_count": self.config.simulation.client_count,
            "rounds": self.config.simulation.rounds,
            "seed": self.config.simulation.seed,
            "local_epochs": self.config.simulation.local_epochs,
        }

    def get_attack_params(self) -> dict:
        """Return the attack parameters."""
        return {
            "enabled": self.config.attack.enabled,
            "scenario": self.config.attack.scenario,
            "attacker_count": self.config.attack.attacker_count,
            "start_round": self.config.attack.start_round,
            "intensity": self.config.attack.intensity,
            "target_clients": self.config.attack.target_clients,
            "rounds": self.config.simulation.rounds,
            "client_count": self.config.simulation.client_count,
        }

    def get_recovery_params(self) -> dict:
        """Return recovery parameters."""
        return {
            "enabled": self.config.recovery.enabled,
            "accuracy_drop_threshold": self.config.recovery.accuracy_drop_threshold,
            "loss_increase_threshold": self.config.recovery.loss_increase_threshold,
        }

    def get_client_ids(self) -> list[str]:
        """Generate deterministic client IDs."""
        return [f"client-{i}" for i in range(self.config.simulation.client_count)]

    def get_fl_config(self) -> dict:
        """Return FL training config for P1."""
        return {
            "local_epochs": self.config.simulation.local_epochs,
            "seed": self.config.simulation.seed,
        }
