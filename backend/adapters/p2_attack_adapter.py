"""Adapter from Person 2's attack manager to the Person 4 runner contract."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from backend.adapters.interfaces import AttackInterface
from core.attacks.attack_manager import AttackManager
from core.models import ModelUpdate
from simulation.scenarios import (
    build_backdoor_scenario,
    build_label_poisoning_scenario,
    build_mixed_attack_scenario,
    build_model_poisoning_scenario,
    build_normal_scenario,
    build_sleeper_scenario,
)


class P2AttackAdapter(AttackInterface):
    """Expose P2's scheduled attacks through P4's update interface."""

    _builders: dict[str, Callable[..., AttackManager]] = {
        "backdoor": build_backdoor_scenario,
        "label_poisoning": build_label_poisoning_scenario,
        "mixed_attack": build_mixed_attack_scenario,
        "model_poisoning": build_model_poisoning_scenario,
        "normal": build_normal_scenario,
        "sleeper": build_sleeper_scenario,
    }

    def __init__(self) -> None:
        self._manager: AttackManager | None = None
        self._client_ids: list[str] = []
        self._scenario_key: tuple[Any, ...] | None = None

    def apply_attacks(self, updates: list[ModelUpdate], round_id: int, config: dict) -> list[ModelUpdate]:
        if not updates or not config.get("enabled", True):
            return updates
        client_ids = list(dict.fromkeys(update.client_id for update in updates))
        self._ensure_manager(client_ids, config)
        assert self._manager is not None
        attacked_updates: list[ModelUpdate] = []
        for update in updates:
            attack = self._manager.get_attack(round_id, update.client_id)
            if attack is None:
                attacked_updates.append(update)
                continue
            original = dict(update.parameters)
            transformed = attack.apply_update_attack({
                name: np.asarray(value, dtype=float)
                for name, value in original.items()
            })
            parameters = {
                name: self._restore_shape(value, original[name])
                for name, value in transformed.items()
            }
            attacked_updates.append(update.model_copy(update={"parameters": parameters}))
        return attacked_updates

    def get_attacker_ids(self, round_id: int, config: dict) -> list[str]:
        if not config.get("enabled", True):
            return []
        if self._manager is None:
            self._ensure_manager(self._client_ids, config)
        assert self._manager is not None
        ground_truth = self._manager.get_ground_truth_for_evaluation()
        return [
            client_id
            for client_id, attack_type in ground_truth.get(round_id, {}).items()
            if attack_type != "honest"
        ]

    def _ensure_manager(self, client_ids: list[str], config: dict) -> None:
        scenario = config.get("scenario", "normal")
        key = (tuple(client_ids), scenario, tuple(sorted(config.items())))
        if self._manager is not None and self._scenario_key == key:
            return
        if scenario not in self._builders:
            raise ValueError(f"Unsupported attack scenario: {scenario}")
        self._client_ids = client_ids
        targets = self._target_clients(client_ids, config)
        rounds = max(int(config.get("total_rounds", config.get("rounds", 10))), 1)
        start_round = int(config.get("start_round", 1))
        builder = self._builders[scenario]
        if scenario == "normal":
            manager = builder(total_rounds=rounds, client_ids=client_ids)
        elif scenario == "mixed_attack":
            split = max(len(targets) // 2, 1)
            manager = builder(
                client_ids=client_ids,
                byzantine_clients=targets[:split],
                backdoor_clients=targets[split:] or targets[:1],
                start_round=start_round,
                total_rounds=rounds,
            )
        elif scenario == "sleeper":
            manager = builder(
                client_ids=client_ids,
                target_client=targets[0],
                stealth_round=max(start_round - 1, 1),
                escalation_round=start_round,
                total_rounds=rounds,
            )
        else:
            manager = builder(
                client_ids=client_ids,
                target_clients=targets,
                start_round=start_round,
                total_rounds=rounds,
            )
        self._manager = manager
        self._scenario_key = key

    @staticmethod
    def _target_clients(client_ids: list[str], config: dict) -> list[str]:
        configured = [client for client in config.get("target_clients", []) if client in client_ids]
        if configured:
            return configured
        count = max(int(config.get("attacker_count", 1)), 1)
        return client_ids[-count:]

    @staticmethod
    def _restore_shape(value: Any, original: Any) -> Any:
        array = np.asarray(value)
        if isinstance(original, list):
            return array.tolist()
        if array.ndim == 0:
            return array.item()
        return array
