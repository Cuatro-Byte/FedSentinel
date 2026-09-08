import torch
from typing import Dict, List


class CheckpointManager:
    """
    Infrastructure component responsible for safely preserving historical global 
    model states in memory.
    
    It guarantees immutability of saved checkpoints and prevents silent overwriting 
    of historical model versions, ensuring reliable recovery capabilities for 
    higher-level orchestrators.
    """

    def __init__(self):
        """Initialize an empty in-memory checkpoint registry."""
        self._checkpoints: Dict[str, Dict[str, torch.Tensor]] = {}
        # Keep an explicit insertion order list for deterministic listing
        self._version_order: List[str] = []

    def save(self, model_version: str, model_state: Dict[str, torch.Tensor]) -> None:
        """
        Save a model state into the checkpoint registry.
        
        Args:
            model_version: A unique string identifier for this version.
            model_state: A dictionary mapping parameter names to PyTorch Tensors.
            
        Raises:
            ValueError: If the version is invalid, already exists, or the state is malformed.
        """
        if not model_version or not isinstance(model_version, str):
            raise ValueError("model_version must be a non-empty string.")
            
        if self.exists(model_version):
            raise ValueError(f"Checkpoint for {model_version} already exists. Historical states cannot be overwritten.")
            
        if not model_state or not isinstance(model_state, dict):
            raise ValueError("model_state must be a non-empty dictionary.")

        safe_state: Dict[str, torch.Tensor] = {}

        for key, tensor in model_state.items():
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Parameter '{key}' is not a PyTorch Tensor.")
                
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                raise ValueError(f"Parameter '{key}' contains NaN or Inf values. Checkpoint rejected.")
                
            # Clone to ensure the stored tensor is entirely decoupled from the caller's memory
            safe_state[key] = tensor.clone().detach()

        self._checkpoints[model_version] = safe_state
        self._version_order.append(model_version)

    def load(self, model_version: str) -> Dict[str, torch.Tensor]:
        """
        Retrieve a safe copy of a historical model state.
        
        Args:
            model_version: The identifier of the checkpoint to load.
            
        Returns:
            A new dictionary mapping parameter names to cloned PyTorch Tensors.
            
        Raises:
            KeyError: If the checkpoint does not exist.
            ValueError: If model_version is invalid.
        """
        if not model_version or not isinstance(model_version, str):
            raise ValueError("model_version must be a non-empty string.")

        if not self.exists(model_version):
            raise KeyError(f"Checkpoint {model_version} not found.")

        stored_state = self._checkpoints[model_version]
        safe_copy: Dict[str, torch.Tensor] = {}

        for key, tensor in stored_state.items():
            # Clone again so the caller cannot mutate our internal checkpoint
            safe_copy[key] = tensor.clone().detach()

        return safe_copy

    def exists(self, model_version: str) -> bool:
        """
        Check if a checkpoint exists for the given version.
        
        Args:
            model_version: The version string to check.
            
        Returns:
            True if the checkpoint exists, False otherwise.
        """
        if not model_version or not isinstance(model_version, str):
            return False
        return model_version in self._checkpoints

    def list_versions(self) -> List[str]:
        """
        List all saved model versions in deterministic (insertion) order.
        
        Returns:
            A new list of version strings.
        """
        return list(self._version_order)
