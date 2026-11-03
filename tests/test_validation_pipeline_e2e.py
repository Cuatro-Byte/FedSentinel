"""End-to-End unified validation pipeline test suite for FedSentinel.

Validates:
- Phase 1: Interface & Defaults (graceful fallbacks, cold-start baselines).
- Phase 2: Contract Invariants & Safety (ground-truth segregation, baseline freezing, memory reset).
- Phase 3: Live Attack Vectors (ScalingAttack applied to ModelUpdate, Canonical Sleeper lifecycle).
"""

import inspect
import sys
import types
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

import core
from core.validation.server_validator import ServerValidationGate

# ---------------------------------------------------------------------------
# Dynamic contract fallback resolution for ModelUpdate & ScalingAttack
# ---------------------------------------------------------------------------

try:
    from core.models.model_update import ModelUpdate  # type: ignore
except (ImportError, ModuleNotFoundError):
    try:
        from core.p1_models.model_update import ModelUpdate  # type: ignore
    except (ImportError, ModuleNotFoundError):
        @dataclass
        class ModelUpdate:  # type: ignore
            """Canonical structure representing a client's model update."""
            update_id: str = "up_001"
            run_id: str = "run_001"
            round_id: int = 1
            client_id: str = "client_1"
            model_version: str = "v1"
            base_model_version: str = "v0"
            parameters: Dict[str, Any] = field(default_factory=dict)
            sample_count: int = 100
            local_loss: Optional[float] = None
            local_accuracy: Optional[float] = None
            training_epochs: int = 1
            learning_rate: Optional[float] = 0.01
            created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
            metadata: Dict[str, Any] = field(default_factory=dict)

        models_mod = getattr(core, "models", types.ModuleType("core.models"))
        models_mod.ModelUpdate = ModelUpdate
        core.models = models_mod
        sys.modules["core.models"] = models_mod
        sys.modules["core.models.model_update"] = models_mod

if "core.attacks.scaling_attack" not in sys.modules:
    try:
        import core.attacks.scaling_attack  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        try:
            from core.attacks.model_poisoning import ScalingAttack as _ScalingAttack
        except (ImportError, ModuleNotFoundError):
            class _ScalingAttack:
                """Update-boosting / gradient scaling attack conforming to Person 2 contract."""

                def __init__(
                    self, client_id: str = "attacker", scale_factor: float = 10.0
                ) -> None:
                    self.client_id = client_id
                    self.scale_factor = float(scale_factor)
                    self.attack_type = "scaling"

                def apply_update_attack(self, delta: Dict[str, Any]) -> Dict[str, Any]:
                    return {k: (v * self.scale_factor) for k, v in delta.items()}

        attacks_mod = getattr(core, "attacks", types.ModuleType("core.attacks"))
        scaling_mod = types.ModuleType("core.attacks.scaling_attack")
        scaling_mod.ScalingAttack = _ScalingAttack
        attacks_mod.scaling_attack = scaling_mod
        core.attacks = attacks_mod
        sys.modules["core.attacks"] = attacks_mod
        sys.modules["core.attacks.scaling_attack"] = scaling_mod


# ===========================================================================
# Phase 1: Interface & Defaults
# ===========================================================================

def test_phase1_fallback_when_none_dataset() -> None:
    """Phase 1: Verify graceful fallback when val_dataset=None."""
    gate = ServerValidationGate(val_dataset=None)
    model = nn.Sequential(nn.Linear(4, 2))

    metrics, loss_spiked = gate.evaluate_checkpoint(model)

    assert loss_spiked is False
    assert metrics["val_loss"] == 0.0
    assert metrics["val_acc"] == 1.0
    assert metrics["loss_delta"] == 0.0
    assert metrics["acc_delta"] == 0.0


def test_phase1_cold_start_baseline_establishment() -> None:
    """Phase 1: Cold-start baseline establishment across clean rounds without false positives."""
    torch.manual_seed(42)
    x = torch.randn(40, 4)
    y = torch.randint(0, 2, (40,))
    dataset = TensorDataset(x, y)
    model = nn.Sequential(nn.Linear(4, 2))

    gate = ServerValidationGate(val_dataset=dataset, batch_size=16)

    # Round 1: Cold-start establishes initial baseline
    metrics_r1, spiked_r1 = gate.evaluate_checkpoint(model)
    assert spiked_r1 is False
    assert metrics_r1["loss_delta"] == 0.0
    assert metrics_r1["acc_delta"] == 0.0
    assert gate.previous_loss == metrics_r1["val_loss"]
    assert gate.previous_acc == metrics_r1["val_acc"]

    # Round 2: Subsequent clean evaluation preserves stability without false alarms
    metrics_r2, spiked_r2 = gate.evaluate_checkpoint(model)
    assert spiked_r2 is False
    assert abs(metrics_r2["loss_delta"]) < 1e-5
    assert abs(metrics_r2["acc_delta"]) < 1e-5
    assert gate.previous_loss == metrics_r2["val_loss"]
    assert gate.previous_acc == metrics_r2["val_acc"]


# ===========================================================================
# Phase 2: Contract Invariants & Safety
# ===========================================================================

def test_phase2_ground_truth_segregation() -> None:
    """Phase 2: Attaching model.is_malicious=True must not leak into metrics or alter gate logic."""
    # Enforce Section 3 invariant: evaluate_checkpoint accepts strictly nn.Module
    sig = inspect.signature(ServerValidationGate.evaluate_checkpoint)
    assert list(sig.parameters.keys()) == ["self", "model"]

    torch.manual_seed(42)
    x = torch.randn(40, 4)
    y = torch.randint(0, 2, (40,))
    dataset = TensorDataset(x, y)
    model = nn.Sequential(nn.Linear(4, 2))

    gate = ServerValidationGate(val_dataset=dataset, batch_size=16)

    # Establish baseline
    metrics_base, spiked_base = gate.evaluate_checkpoint(model)
    assert spiked_base is False

    # Attach adversarial ground-truth metadata
    setattr(model, "is_malicious", True)

    metrics_test, spiked_test = gate.evaluate_checkpoint(model)

    # Evaluation must evaluate raw tensors only: clean weights pass regardless of metadata
    assert spiked_test is False
    assert "is_malicious" not in metrics_test


def test_phase2_baseline_freezing_on_loss_spike() -> None:
    """Phase 2: Baseline freezing - on loss spike, previous_loss stays frozen at reference value."""
    torch.manual_seed(42)
    x = torch.randn(40, 4)
    y = torch.randint(0, 2, (40,))
    dataset = TensorDataset(x, y)
    model = nn.Sequential(nn.Linear(4, 2))

    gate = ServerValidationGate(val_dataset=dataset, batch_size=16)

    # Clean round establishes baseline
    metrics_clean, clean_spiked = gate.evaluate_checkpoint(model)
    assert clean_spiked is False
    reference_loss = gate.previous_loss
    reference_acc = gate.previous_acc
    assert reference_loss is not None
    assert reference_acc is not None

    # Corrupt model weights causing degradation
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(25.0)

    metrics_corrupt, loss_spiked = gate.evaluate_checkpoint(model)
    assert loss_spiked is True
    assert metrics_corrupt["loss_delta"] >= 0.25

    # Crucial assertion: gate MUST freeze baselines at clean reference values
    assert gate.previous_loss == reference_loss
    assert gate.previous_acc == reference_acc
    assert gate.previous_loss != metrics_corrupt["val_loss"]


def test_phase2_memory_reset_via_reset_baseline() -> None:
    """Phase 2: Memory reset via reset_baseline() resets previous loss and accuracy to None."""
    torch.manual_seed(42)
    x = torch.randn(20, 4)
    y = torch.randint(0, 2, (20,))
    dataset = TensorDataset(x, y)
    model = nn.Sequential(nn.Linear(4, 2))

    gate = ServerValidationGate(val_dataset=dataset)
    gate.evaluate_checkpoint(model)
    assert gate.previous_loss is not None
    assert gate.previous_acc is not None

    gate.reset_baseline()
    assert gate.previous_loss is None
    assert gate.previous_acc is None


# ===========================================================================
# Phase 3: Live Attack Vectors
# ===========================================================================

def test_phase3_scaling_attack_on_model_update() -> None:
    """Phase 3: Integration with Person 2's ScalingAttack(scale_factor=20.0) applied to ModelUpdate."""
    from core.attacks.scaling_attack import ScalingAttack

    torch.manual_seed(42)
    x = torch.randn(40, 4)
    y = torch.randint(0, 2, (40,))
    dataset = TensorDataset(x, y)
    model = nn.Sequential(nn.Linear(4, 2))

    gate = ServerValidationGate(val_dataset=dataset, batch_size=16)

    # Establish baseline
    base_metrics, base_spiked = gate.evaluate_checkpoint(model)
    assert base_spiked is False

    # Simulate gradient update
    criterion = nn.CrossEntropyLoss()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()

    param_delta = {
        name: param.grad.detach().cpu().numpy()
        for name, param in model.named_parameters()
        if param.grad is not None
    }

    # Encapsulate update in ModelUpdate
    update = ModelUpdate(
        update_id="up_malicious_scaling",
        client_id="attacker_p2",
        parameters=param_delta,
        run_id="run_1",
        round_id=1,
        model_version="1",
        base_model_version="0",
        sample_count=40,
        training_epochs=1,
    )

    # Person 2's ScalingAttack with scale_factor=20.0
    attacker = ScalingAttack(client_id=update.client_id, scale_factor=20.0)
    poisoned_delta = attacker.apply_update_attack(update.parameters)

    # Apply poisoned update to model
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in poisoned_delta:
                param.add_(
                    torch.as_tensor(
                        poisoned_delta[name],
                        dtype=param.dtype,
                        device=param.device,
                    )
                )

    metrics, loss_spiked = gate.evaluate_checkpoint(model)
    assert loss_spiked is True
    assert metrics["loss_delta"] >= gate.loss_spike_threshold


def test_phase3_canonical_sleeper_lifecycle_simulation() -> None:
    """Phase 3: Canonical Sleeper lifecycle simulation across 4 rounds.

    - Rounds 1-2: Honest training (quiet, no alarm).
    - Round 3: Covert perturbation (sub-threshold perturbation, quiet).
    - Round 4: Escalation (high-intensity attack, trips loss_spiked=True).
    """
    torch.manual_seed(42)
    x = torch.randn(60, 4)
    y = torch.randint(0, 2, (60,))
    dataset = TensorDataset(x, y)
    model = nn.Sequential(nn.Linear(4, 2))

    gate = ServerValidationGate(val_dataset=dataset, batch_size=16)

    # --- Round 1: Honest (Cold-start baseline) ---
    m1, s1 = gate.evaluate_checkpoint(model)
    assert s1 is False
    assert m1["loss_delta"] == 0.0

    # --- Round 2: Honest (Legitimate model training) ---
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss = nn.CrossEntropyLoss()(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    m2, s2 = gate.evaluate_checkpoint(model)
    assert s2 is False
    assert gate.previous_loss == m2["val_loss"]

    # --- Round 3: Covert Perturbation (Sub-threshold attack, stealth) ---
    # Attacker injects small covert noise that stays below spike thresholds
    with torch.no_grad():
        for param in model.parameters():
            param.add_(0.01 * torch.randn_like(param))

    m3, s3 = gate.evaluate_checkpoint(model)
    # Must remain quiet (sub-threshold)
    assert s3 is False
    assert abs(m3["loss_delta"]) < gate.loss_spike_threshold
    assert gate.previous_loss == m3["val_loss"]

    # --- Round 4: Escalation (High-intensity attack) ---
    # Sleeper agent escalates, injecting dominant attack that overwhelms model
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(20.0)

    m4, s4 = gate.evaluate_checkpoint(model)
    # Must trip validation alarm
    assert s4 is True
    assert m4["loss_delta"] >= gate.loss_spike_threshold
    # Baseline must remain frozen at Round 3 reference level
    assert gate.previous_loss == m3["val_loss"]
