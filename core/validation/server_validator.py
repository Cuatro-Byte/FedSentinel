"""Server validation gate implementation for FedSentinel.

Enforces the Ground-Truth Segregation Invariant:
The validator evaluates raw tensor forward passes over a validation dataset
and MUST NEVER access, inspect, or accept ground-truth attack or client flags
(e.g., `is_malicious`).
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class ServerValidationGate:
    """Validation gate for evaluating aggregated global models on the server.

    Monitors validation loss spikes and accuracy drops across federated learning rounds
    to identify potential poisoning or degradation without accessing client-side metadata
    or ground-truth attack labels.
    """

    def __init__(
        self,
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 32,
        loss_spike_threshold: float = 0.35,
        acc_drop_threshold: float = 0.15,
        device: str = "cpu",
    ) -> None:
        """Initialize the server validation gate.

        Args:
            val_dataset: Optional PyTorch Dataset used for validation. If None, evaluation
                is bypassed and default metrics are returned.
            batch_size: Batch size for validation DataLoader.
            loss_spike_threshold: Threshold for loss increase that triggers a validation alarm.
            acc_drop_threshold: Threshold for accuracy drop that triggers a validation alarm.
            device: Target device string (e.g., 'cpu', 'cuda') on which to perform evaluation.
        """
        self.val_dataset: Optional[Dataset] = val_dataset
        self.batch_size: int = batch_size
        self.loss_spike_threshold: float = loss_spike_threshold
        self.acc_drop_threshold: float = acc_drop_threshold
        self.device: torch.device = torch.device(device)

        self.previous_loss: Optional[float] = None
        self.previous_acc: Optional[float] = None
        self.criterion: nn.Module = nn.CrossEntropyLoss()

        self.val_loader: Optional[DataLoader] = None
        if self.val_dataset is not None:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
            )

    def evaluate_checkpoint(
        self, model: nn.Module
    ) -> Tuple[Dict[str, float], bool]:
        """Evaluate a global model checkpoint on the server validation dataset.

        Evaluates the model in evaluation mode under `torch.no_grad()`, strictly ensuring
        zero mutation of model weights and internal training states.

        Args:
            model: PyTorch model module to evaluate.

        Returns:
            Tuple of:
                - metrics_dict: Dictionary containing:
                    * "val_loss": Validation loss over dataset.
                    * "val_acc": Validation classification accuracy [0.0, 1.0].
                    * "loss_delta": Change in loss compared to previous baseline.
                    * "acc_delta": Drop in accuracy compared to previous baseline.
                - loss_spiked: Boolean indicating whether validation degradation exceeded
                  configured thresholds.
        """
        if self.val_dataset is None or self.val_loader is None:
            return (
                {
                    "val_loss": 0.0,
                    "val_acc": 1.0,
                    "loss_delta": 0.0,
                    "acc_delta": 0.0,
                },
                False,
            )

        # Preserve original training mode to avoid unintended side effects
        original_mode = model.training
        model.eval()

        total_loss: float = 0.0
        correct: int = 0
        total_samples: int = 0

        try:
            with torch.no_grad():
                for batch in self.val_loader:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = model(inputs)
                    loss = self.criterion(outputs, targets)

                    total_loss += loss.item() * inputs.size(0)
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == targets).sum().item()
                    total_samples += targets.size(0)
        finally:
            model.train(original_mode)

        val_loss: float = (
            total_loss / total_samples if total_samples > 0 else 0.0
        )
        val_acc: float = (
            float(correct) / total_samples if total_samples > 0 else 0.0
        )

        # Compute deltas against previous baseline if one exists
        if self.previous_loss is not None and self.previous_acc is not None:
            loss_delta = val_loss - self.previous_loss
            acc_delta = self.previous_acc - val_acc
        else:
            loss_delta = 0.0
            acc_delta = 0.0

        loss_spiked: bool = bool(
            (loss_delta >= self.loss_spike_threshold)
            or (acc_delta >= self.acc_drop_threshold)
        )

        # Update baseline only when no spike / degradation is detected
        if not loss_spiked:
            self.previous_loss = val_loss
            self.previous_acc = val_acc

        metrics_dict: Dict[str, float] = {
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "loss_delta": float(loss_delta),
            "acc_delta": float(acc_delta),
        }

        return metrics_dict, loss_spiked

    def reset_baseline(self) -> None:
        """Reset historical baseline loss and accuracy trackers."""
        self.previous_loss = None
        self.previous_acc = None
