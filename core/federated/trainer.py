import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class TrainingResult:
    """
    Structure representing the result of a local training session.
    This is NOT a ModelUpdate. The Client uses this output to generate
    the canonical ModelUpdate.
    """
    model_state: Dict[str, Any]
    loss: float
    accuracy: float
    sample_count: int
    epochs_completed: int


class LocalTrainer:
    """
    A reusable local training component for standard supervised federated learning.
    It isolates training logic from the client structure, aggregation, and attack mechanisms.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 1,
        device: str = "cpu"
    ):
        """
        Initialize the trainer with basic configuration.
        
        Args:
            learning_rate: Learning rate for the optimizer.
            epochs: Number of epochs to train locally.
            device: Device to use for training ("cpu" or "cuda").
        """
        if learning_rate <= 0:
            raise ValueError("Learning rate must be strictly positive.")
        if epochs <= 0:
            raise ValueError("Epochs must be strictly positive.")

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device(device)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, model: nn.Module, dataloader: DataLoader) -> TrainingResult:
        """
        Execute local supervised training for the configured number of epochs.
        
        Args:
            model: PyTorch neural network to train.
            dataloader: DataLoader providing the local dataset.
            
        Returns:
            TrainingResult containing the trained state and metrics.
        """
        if model is None:
            raise ValueError("A valid PyTorch model must be provided.")
        if len(dataloader.dataset) == 0:
            raise ValueError("Cannot train on an empty dataset.")

        model = model.to(self.device)
        model.train()
        
        # We use a standard SGD optimizer for prototypical FL
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Track metric counts to compute averages
        batch_count = 0

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for batch in dataloader:
                if not isinstance(batch, (list, tuple)) or len(batch) != 2:
                    raise ValueError(f"Malformed batch: expected (data, target), got {type(batch).__name__}.")
                
                data, target = batch
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                output = model(data)
                loss = self.criterion(output, target)
                
                loss.backward()
                optimizer.step()
                
                # Accumulate metrics
                batch_size = data.size(0)
                epoch_loss += loss.item() * batch_size
                
                # Calculate accuracy
                _, predicted = torch.max(output.data, 1)
                epoch_correct += (predicted == target).sum().item()
                epoch_samples += batch_size
                
                batch_count += 1
            
            # Since we return a final metric summarizing the whole training,
            # we accumulate across all epochs.
            total_loss += epoch_loss
            correct_predictions += epoch_correct
            total_samples += epoch_samples

        # Calculate final averaged metrics across all seen batches/epochs
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        avg_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

        # We must return model state strictly on CPU
        model_state = {k: v.cpu() for k, v in model.state_dict().items()}

        return TrainingResult(
            model_state=model_state,
            loss=avg_loss,
            accuracy=avg_accuracy,
            sample_count=len(dataloader.dataset),
            epochs_completed=self.epochs
        )
