"""Isolated unit tests for ServerValidationGate conforming to Team Engineering Contract v2.0."""

import inspect
import sys
import types
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

import core
from core.validation.server_validator import ServerValidationGate

# Ensure core.attacks.scaling_attack is importable in isolated test environments
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

                def apply_update_attack(self, delta: dict) -> dict:
                    return {k: (v * self.scale_factor) for k, v in delta.items()}

        attacks_mod = getattr(core, "attacks", types.ModuleType("core.attacks"))
        scaling_mod = types.ModuleType("core.attacks.scaling_attack")
        scaling_mod.ScalingAttack = _ScalingAttack
        attacks_mod.scaling_attack = scaling_mod
        core.attacks = attacks_mod
        sys.modules["core.attacks"] = attacks_mod
        sys.modules["core.attacks.scaling_attack"] = scaling_mod


def test_validator_baseline_progression() -> None:
    """Verify initial evaluation establishes baseline and subsequent clean passes maintain stability."""
    torch.manual_seed(42)
    # Synthetic TensorDataset: 40 samples, 4 features, 2 classes
    x = torch.randn(40, 4)
    y = torch.randint(0, 2, (40,))
    dataset = TensorDataset(x, y)
    model = nn.Sequential(nn.Linear(4, 2))

    validator = ServerValidationGate(val_dataset=dataset, batch_size=16)

    # Initial pass establishes baseline
    metrics1, loss_spiked1 = validator.evaluate_checkpoint(model)
    assert loss_spiked1 is False
    assert metrics1["loss_delta"] == 0.0
    assert metrics1["acc_delta"] == 0.0
    assert validator.previous_loss is not None
    assert validator.previous_acc is not None
    assert metrics1["val_loss"] == validator.previous_loss
    assert metrics1["val_acc"] == validator.previous_acc

    # Second pass with clean model
    metrics2, loss_spiked2 = validator.evaluate_checkpoint(model)
    assert loss_spiked2 is False
    assert abs(metrics2["loss_delta"]) < 1e-5
    assert abs(metrics2["acc_delta"]) < 1e-5


def test_validator_detects_adversarial_degradation() -> None:
    """Verify corrupted/poisoned model weights trigger validation loss spike."""
    torch.manual_seed(42)
    x = torch.randn(40, 4)
    y = torch.randint(0, 2, (40,))
    dataset = TensorDataset(x, y)
    model = nn.Sequential(nn.Linear(4, 2))

    validator = ServerValidationGate(val_dataset=dataset, batch_size=16)

    # Initialize baseline with clean model
    metrics_clean, clean_spiked = validator.evaluate_checkpoint(model)
    assert clean_spiked is False
    baseline_loss = validator.previous_loss
    baseline_acc = validator.previous_acc

    # Corrupt model weights (multiply parameters by 25.0 to simulate model poisoning)
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(25.0)

    # Evaluate corrupted model
    metrics_corrupt, loss_spiked = validator.evaluate_checkpoint(model)
    assert loss_spiked is True
    assert metrics_corrupt["loss_delta"] >= 0.25

    # Assert baseline was not corrupted or overwritten during spike
    assert validator.previous_loss == baseline_loss
    assert validator.previous_acc == baseline_acc


def test_validator_handles_none_dataset_gracefully() -> None:
    """Verify validator handles None dataset without error returning default metrics."""
    validator = ServerValidationGate(val_dataset=None)
    model = nn.Sequential(nn.Linear(4, 2))

    metrics, loss_spiked = validator.evaluate_checkpoint(model)
    assert loss_spiked is False
    assert metrics["val_loss"] == 0.0
    assert metrics["val_acc"] == 1.0
    assert metrics["loss_delta"] == 0.0
    assert metrics["acc_delta"] == 0.0


def test_validator_reset_baseline() -> None:
    """Verify reset_baseline() resets historical baselines back to None."""
    torch.manual_seed(42)
    x = torch.randn(20, 4)
    y = torch.randint(0, 2, (20,))
    dataset = TensorDataset(x, y)
    model = nn.Sequential(nn.Linear(4, 2))

    validator = ServerValidationGate(val_dataset=dataset)
    metrics, loss_spiked = validator.evaluate_checkpoint(model)
    assert validator.previous_loss is not None
    assert validator.previous_acc is not None

    validator.reset_baseline()
    assert validator.previous_loss is None
    assert validator.previous_acc is None


def test_validator_no_ground_truth_leakage() -> None:
    """Verify Section 3 invariant: evaluate_checkpoint strictly evaluates nn.Module without accepting or inspecting is_malicious."""
    # 1. Verify signature accepts strictly model
    sig = inspect.signature(ServerValidationGate.evaluate_checkpoint)
    param_names = list(sig.parameters.keys())
    assert param_names == ["self", "model"]
    assert "is_malicious" not in param_names

    torch.manual_seed(42)
    x = torch.randn(40, 4)
    y = torch.randint(0, 2, (40,))
    dataset = TensorDataset(x, y)
    validator = ServerValidationGate(val_dataset=dataset)

    # 2. Attach adversarial ground-truth metadata to model
    model = nn.Sequential(nn.Linear(4, 2))
    setattr(model, "is_malicious", True)

    metrics, loss_spiked = validator.evaluate_checkpoint(model)

    # 3. Verify evaluate_checkpoint does not inspect or act on the flag (clean weights still pass)
    assert loss_spiked is False
    assert "is_malicious" not in metrics


def test_validator_catches_scaling_attack_shock() -> None:
    """Verify that ServerValidationGate correctly triggers when attacked by Person 2's ScalingAttack."""
    # Import ScalingAttack from core.attacks.scaling_attack (owned by Person 2)
    from core.attacks.scaling_attack import ScalingAttack

    torch.manual_seed(42)
    # Instantiate small model, synthetic dataset, and ServerValidationGate
    x = torch.randn(40, 4)
    y = torch.randint(0, 2, (40,))
    dataset = TensorDataset(x, y)
    model = nn.Sequential(nn.Linear(4, 2))

    validator = ServerValidationGate(val_dataset=dataset, batch_size=16)

    # Establish baseline loss
    base_metrics, base_spiked = validator.evaluate_checkpoint(model)
    assert base_spiked is False
    assert validator.previous_loss is not None

    # Simulate a poisoned gradient step with scale_factor=15.0
    criterion = nn.CrossEntropyLoss()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()

    # Raw delta in adversarial gradient ascent direction
    raw_delta = {
        name: param.grad.detach().cpu().numpy()
        for name, param in model.named_parameters()
        if param.grad is not None
    }

    attacker = ScalingAttack(client_id="attacker_p2", scale_factor=15.0)
    scaled_delta = attacker.apply_update_attack(raw_delta)

    # Apply the scaled delta to the model parameters
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in scaled_delta:
                param.add_(
                    torch.as_tensor(
                        scaled_delta[name],
                        dtype=param.dtype,
                        device=param.device,
                    )
                )

    # Run evaluate_checkpoint and assert validation gate catches adversarial shock
    metrics, loss_spiked = validator.evaluate_checkpoint(model)
    assert loss_spiked is True
