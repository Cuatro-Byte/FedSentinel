import uuid
import copy
from datetime import datetime
import torch.nn as nn
from torch.utils.data import DataLoader

from core.p1_models.model_update import ModelUpdate
from core.federated.trainer import LocalTrainer


class SimulatedClient:
    """
    Represents a single federated learning client in the simulation.
    Orchestrates the local training process using a LocalTrainer and 
    generates a canonical ModelUpdate.
    """

    def __init__(self, client_id: str, dataloader: DataLoader, trainer: LocalTrainer):
        """
        Initialize the simulated client.
        
        Args:
            client_id: Unique identifier for this client.
            dataloader: PyTorch DataLoader containing the client's local dataset.
            trainer: LocalTrainer instance configured for this client.
        """
        if not client_id or not isinstance(client_id, str):
            raise ValueError("client_id must be a non-empty string.")
        if dataloader is None or len(dataloader.dataset) == 0:
            raise ValueError("A valid non-empty dataloader is required.")
        if trainer is None or not isinstance(trainer, LocalTrainer):
            raise ValueError("A valid LocalTrainer instance is required.")

        self.client_id = client_id
        self.dataloader = dataloader
        self.trainer = trainer

    def train(
        self,
        global_model: nn.Module,
        run_id: str,
        round_id: int,
        base_model_version: str,
        model_version: str
    ) -> ModelUpdate:
        """
        Execute local training on a copy of the global model and return a ModelUpdate.
        
        Args:
            global_model: The current global PyTorch model.
            run_id: Identifier for the current simulation run.
            round_id: Current federated learning round number.
            base_model_version: Version of the global model received.
            model_version: Version of the resulting local model.
            
        Returns:
            A canonical ModelUpdate object containing the trained parameters and metrics.
        """
        if global_model is None or not isinstance(global_model, nn.Module):
            raise ValueError("A valid PyTorch model must be provided.")
        if not run_id or not isinstance(run_id, str):
            raise ValueError("run_id must be a non-empty string.")
        if round_id < 0:
            raise ValueError("round_id must be a non-negative integer.")
        if not base_model_version or not isinstance(base_model_version, str):
            raise ValueError("base_model_version must be a non-empty string.")
        if not model_version or not isinstance(model_version, str):
            raise ValueError("model_version must be a non-empty string.")

        # Create a deep copy of the model to ensure local training 
        # doesn't accidentally mutate the server's global model object.
        local_model = copy.deepcopy(global_model)

        # Delegate the actual training to the LocalTrainer
        training_result = self.trainer.train(local_model, self.dataloader)

        # Generate a unique update ID
        update_id = str(uuid.uuid4())

        # Compute parameter deltas: W_local - W_base
        base_state = {k: v.cpu() for k, v in global_model.state_dict().items()}
        local_state = training_result.model_state
        delta = {}
        
        for k, v_local in local_state.items():
            v_base = base_state[k]
            if v_local.is_floating_point():
                delta[k] = v_local - v_base
            else:
                delta[k] = v_local.to(torch.float32) - v_base.to(torch.float32)

        # Construct and return the canonical ModelUpdate
        return ModelUpdate(
            update_id=update_id,
            run_id=run_id,
            round_id=round_id,
            client_id=self.client_id,
            model_version=model_version,
            base_model_version=base_model_version,
            parameters=delta,
            sample_count=training_result.sample_count,
            local_loss=training_result.loss,
            local_accuracy=training_result.accuracy,
            training_epochs=training_result.epochs_completed,
            learning_rate=self.trainer.learning_rate,
            created_at=datetime.now(),
            metadata={"optimizer": "SGD"} # minimal, neutral metadata
        )
