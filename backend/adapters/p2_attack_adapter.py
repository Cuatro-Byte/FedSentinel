"""Adapter from Person 2's attack manager to the Person 4 runner contract."""

from __future__ import annotations

from typing import Any, Callable
import random

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
    build_sybil_scenario,
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
        "sybil": build_sybil_scenario,
    }

    def __init__(self) -> None:
        self._manager: AttackManager | None = None
        self._client_ids: list[str] = []
        self._scenario_key: tuple[Any, ...] | None = None

    def apply_data_attack(
        self,
        client_id: str,
        dataset: Any,
        round_id: int,
        config: dict,
        client_ids: list[str] | None = None,
    ) -> Any:
        """Apply data-level attack to a client's dataset before local training."""
        if not config.get("enabled", True):
            return dataset
        c_ids = client_ids or self._client_ids
        if not c_ids:
            c_ids = [client_id]
        self._ensure_manager(c_ids, config)
        assert self._manager is not None
        attack = self._manager.get_attack(round_id, client_id)
        if attack is None:
            return dataset
        return attack.apply_data_attack(dataset)

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
        raw_scenario = config.get("scenario", "normal")
        scenario = "label_poisoning" if raw_scenario == "label_flipping" else raw_scenario
        key = (tuple(client_ids), scenario, tuple(sorted((k, str(v)) for k, v in config.items())))
        if self._manager is not None and self._scenario_key == key:
            return
        if scenario not in self._builders:
            raise ValueError(f"Unsupported attack scenario: {scenario}")
        if not client_ids:
            client_count = int(config.get("client_count", 0))
            if client_count > 0:
                client_ids = [f"client-{i}" for i in range(client_count)]
            elif config.get("target_clients"):
                client_ids = list(config.get("target_clients", []))
        self._client_ids = client_ids
        targets = self._target_clients(client_ids, config)
        rounds = max(int(config.get("total_rounds", config.get("rounds", 10))), 1)
        start_round = int(config.get("start_round", 1))
        intensity = config.get("intensity")
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
            target = targets[0] if targets else client_ids[0]
            manager = builder(
                client_ids=client_ids,
                target_client=target,
                stealth_round=max(start_round - 1, 1),
                escalation_round=start_round,
                total_rounds=rounds,
            )
        elif scenario == "backdoor":
            kwargs: dict[str, Any] = {
                "client_ids": client_ids,
                "target_clients": targets or [client_ids[-1]],
                "start_round": start_round,
                "total_rounds": rounds,
            }
            if intensity is not None:
                kwargs["intensity"] = float(intensity)
            # Check target config if provided
            raw_targets = config.get("targets")
            if isinstance(raw_targets, dict) and "target_label" in raw_targets:
                kwargs["target_label"] = int(raw_targets["target_label"])
            elif isinstance(raw_targets, int):
                kwargs["target_label"] = raw_targets
            manager = builder(**kwargs)
        elif scenario == "model_poisoning":
            kwargs: dict[str, Any] = {
                "client_ids": client_ids,
                "target_clients": targets or ([client_ids[-1]] if client_ids else []),
                "start_round": start_round,
                "total_rounds": rounds,
            }
            if intensity is not None:
                kwargs["scale_factor"] = float(intensity)
            manager = builder(**kwargs)
        elif scenario == "label_poisoning":
            kwargs: dict[str, Any] = {
                "client_ids": client_ids,
                "target_clients": targets or [client_ids[-1]],
                "start_round": start_round,
                "total_rounds": rounds,
            }
            if intensity is not None:
                kwargs["poison_rate"] = min(float(intensity), 1.0)
            raw_targets = config.get("targets")
            if isinstance(raw_targets, dict):
                if "source_class" in raw_targets:
                    kwargs["source_class"] = int(raw_targets["source_class"])
                if "target_class" in raw_targets:
                    kwargs["target_class"] = int(raw_targets["target_class"])
            manager = builder(**kwargs)
        elif scenario == "sybil":
            sybil_targets = targets
            if len(sybil_targets) < 2 and len(client_ids) >= 2:
                sybil_targets = client_ids[-2:]
            kwargs: dict[str, Any] = {
                "client_ids": client_ids,
                "target_clients": sybil_targets,
                "start_round": start_round,
                "total_rounds": rounds,
            }
            if intensity is not None:
                kwargs["scale"] = float(intensity)
            if "scale" in config:
                kwargs["scale"] = float(config["scale"])
            if "shared_seed" in config:
                kwargs["shared_seed"] = int(config["shared_seed"])
            if "target_bias" in config:
                kwargs["target_bias"] = float(config["target_bias"])
            manager = builder(**kwargs)
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
        raw_targets = config.get("targets")
        if not configured and isinstance(raw_targets, list):
            configured = [c for c in raw_targets if isinstance(c, str) and c in client_ids]
        if configured:
            return configured

        count = int(config.get("attacker_count", 1))
        count = max(0, min(count, len(client_ids)))
        if count == 0:
            return []

        seed = config.get("seed")
        if seed is not None:
            rng = random.Random(seed)
            shuffled = list(client_ids)
            rng.shuffle(shuffled)
            return sorted(shuffled[:count])

        return client_ids[-count:]

    @staticmethod
    def _restore_shape(value: Any, original: Any) -> Any:
        array = np.asarray(value)
        if isinstance(original, list):
            return array.tolist()
        if type(original).__name__ == "Tensor":
            import torch
            return torch.from_numpy(array).to(dtype=original.dtype, device=original.device)
        if array.ndim == 0:
            return array.item()
        return array
