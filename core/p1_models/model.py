import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict


class SimpleCNN(nn.Module):
    """
    A lightweight CNN architecture for federated learning simulations.
    Designed to be small enough for fast local training and rapid round execution,
    but complex enough to support meaningful classification tasks.
    
    Default configuration is designed for 28x28 grayscale images (e.g., MNIST/FMNIST).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        # Feature extraction
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Classifier
        # Input size: 28x28 -> pool -> 14x14 -> pool -> 7x7
        # 32 channels * 7 * 7 = 1568
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_weights(self) -> Dict[str, Any]:
        """
        Extract the current model parameters as a dictionary.
        This provides a clean utility for P1 to capture model state 
        for model updates or checkpoints.
        
        Returns:
            Dict[str, Any]: The state dictionary containing the model weights on CPU.
        """
        # Always move weights to CPU before sharing to avoid device mismatches
        # during aggregation or client transfer.
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights: Dict[str, Any]) -> None:
        """
        Load a given set of weights into the model.
        This provides a clean utility for P1 to apply aggregated weights
        or restore checkpoints.
        
        Args:
            weights (Dict[str, Any]): The state dictionary to load.
        """
        self.load_state_dict(weights)
