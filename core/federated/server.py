import copy
from dataclasses import dataclass
from typing import Callable, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.p1_models.model_update import ModelUpdate
from core.p1_models.detection_result import DetectionResult
from core.federated.evaluator import EvaluationResult, Evaluator
from core.federated.client_manager import ClientManager
from core.federated.aggregator import Aggregator
from core.federated.checkpoint import CheckpointManager


@dataclass
class RoundResult:
    """
    Structure representing the result of a completed federated learning round.
    It contains the collected updates and the post-aggregation evaluation metrics.
    
    This does NOT contain threat scores, attack methodologies, or Sentinel flags.
    """
    round_id: int
    model_version: str
    updates: List[ModelUpdate]
    evaluation_result: EvaluationResult
    decisions: Optional[List[DetectionResult]] = None


class Server:
    """
    Orchestration layer for the P1 federated learning infrastructure.
    It coordinates the global model, client selection, local training, 
    aggregation, and evaluation to execute baseline FL rounds safely.
    """

    def __init__(
        self,
        global_model: nn.Module,
        client_manager: ClientManager,
        aggregator: Aggregator,
        evaluator: Evaluator,
        run_id: str,
        initial_model_version: str = "model_v0",
        eval_dataloader: Optional[DataLoader] = None,
        checkpoint_manager: Optional[CheckpointManager] = None
    ):
        """
        Initialize the federated learning server.
        
        Args:
            global_model: The PyTorch neural network serving as the global model.
            client_manager: The component managing the registered simulated clients.
            aggregator: The component performing mathematical update aggregation.
            evaluator: The component responsible for model evaluation.
            run_id: Constant unique identifier for this simulation run.
            initial_model_version: Identifier for the starting model state.
            eval_dataloader: Optional evaluation dataset for the global model.
            checkpoint_manager: Storage component for model versions. Created if None.
            
        Raises:
            ValueError: If any component is missing or invalid.
        """
        if not global_model or not isinstance(global_model, nn.Module):
            raise ValueError("A valid PyTorch global model is required.")
        if not client_manager or not isinstance(client_manager, ClientManager):
            raise ValueError("A valid ClientManager is required.")
        if not aggregator or not isinstance(aggregator, Aggregator):
            raise ValueError("A valid Aggregator is required.")
        if not evaluator or not isinstance(evaluator, Evaluator):
            raise ValueError("A valid Evaluator is required.")
        if not run_id or not isinstance(run_id, str):
            raise ValueError("A valid non-empty run_id is required.")

        self.global_model = global_model
        self.client_manager = client_manager
        self.aggregator = aggregator
        self.evaluator = evaluator
        self.run_id = run_id
        
        self.current_model_version = initial_model_version
        self.current_round = 0
        self.eval_dataloader = eval_dataloader
        
        # Use provided checkpoint manager or instantiate a default in-memory one
        self.checkpoint_manager = checkpoint_manager if checkpoint_manager else CheckpointManager()
        
        # Round history kept in memory for auditing/recovery
        self.history: List[RoundResult] = []

        # Immediately snapshot the initial model state
        self.checkpoint_manager.save(self.current_model_version, self.global_model.state_dict())

    def execute_round(self, num_clients: int, seed: Optional[int] = None) -> RoundResult:
        """
        Execute a single standard federated learning round (baseline FedAvg).
        
        This is the normal path. It uses standard sample-count weighted FedAvg
        and does not require any Sentinel/security decisions.
        
        Args:
            num_clients: The number of clients to participate in this round.
            seed: Optional seed for deterministic client sampling.
            
        Returns:
            A RoundResult object detailing the outcome of the round.
            
        Raises:
            ValueError: If inputs are invalid or empty updates occur.
            RuntimeError: If local training fails.
        """
        return self._run_round(num_clients=num_clients, seed=seed, decisions=None, down_weight_factor=0.5)

    def execute_round_with_decisions(
        self,
        num_clients: int,
        decisions: List[DetectionResult],
        seed: Optional[int] = None,
        down_weight_factor: float = 0.5,
    ) -> RoundResult:
        """
        Execute a security-aware federated learning round using external DetectionResult decisions.

        The Server does not calculate threat scores. It receives pre-computed canonical
        DetectionResult objects from P3 and uses their action field to determine
        aggregation contribution for each client update:

            ACCEPT      → full sample-count weight (1.0)
            DOWN_WEIGHT → down_weight_factor × sample-count weight (default 0.5)
            QUARANTINE  → excluded (0.0)

        Normal FL rounds without security decisions must use execute_round() instead.

        Args:
            num_clients: The number of clients to participate in this round.
            decisions: Pre-computed DetectionResult objects from P3, one per participating client.
            seed: Optional seed for deterministic client sampling.
            down_weight_factor: Float in (0, 1) applied to DOWN_WEIGHT updates.

        Returns:
            A RoundResult object detailing the outcome of the round.

        Raises:
            ValueError: If inputs are invalid, decisions are missing/mismatched, or all clients are quarantined.
            RuntimeError: If local training or evaluation fails.
        """
        if not decisions:
            raise ValueError("execute_round_with_decisions: decisions list cannot be empty.")
        return self._run_round(
            num_clients=num_clients,
            seed=seed,
            decisions=decisions,
            down_weight_factor=down_weight_factor,
        )

    def _run_round(
        self,
        num_clients: int,
        seed: Optional[int],
        decisions: Optional[List[DetectionResult]],
        down_weight_factor: float,
    ) -> RoundResult:
        """Internal shared round execution. Handles both baseline and security-aware paths."""
        if num_clients <= 0:
            raise ValueError("num_clients must be greater than zero.")
            
        if self.eval_dataloader is None:
            raise ValueError("Server must have an eval_dataloader configured to execute a round.")

        # 0. Snapshot current safe state to allow rollback on downstream failure
        previous_state = {k: v.clone() for k, v in self.global_model.state_dict().items()}

        # 1. Determine target round ID and select clients
        target_round = self.current_round + 1
        target_version = f"model_v{target_round}"
        selected_clients = self.client_manager.sample_clients(num_clients, seed)

        # 2. Local Training phase
        updates: List[ModelUpdate] = []
        for client in selected_clients:
            try:
                update = client.train(
                    global_model=self.global_model,
                    run_id=self.run_id,
                    round_id=target_round,
                    base_model_version=self.current_model_version,
                    model_version=f"{self.current_model_version}_{client.client_id}"
                )
                updates.append(update)
            except Exception as e:
                # Explicitly fail the round rather than silently dropping a client
                raise RuntimeError(f"Round failed due to training failure on {client.client_id}: {e}")

        if not updates:
            raise ValueError("Round generated no valid updates.")

        # Validate that updates correspond to the exact global model state before aggregation
        for u in updates:
            if u.base_model_version != self.current_model_version:
                raise ValueError(
                    f"Stale update detected from {u.client_id}: "
                    f"Expected {self.current_model_version}, got {u.base_model_version}."
                )

        # 3. Aggregation phase
        # When decisions are provided, use trust-aware aggregation; otherwise standard FedAvg.
        if decisions is not None:
            new_params = self.aggregator.aggregate_with_decisions(
                updates, decisions, down_weight_factor=down_weight_factor
            )
        else:
            new_params = self.aggregator.aggregate(updates)

        # 4. Global model update and Evaluation phase
        # Reconstruct the candidate model: base_state + aggregated_delta
        new_state = {}
        for k, v in previous_state.items():
            delta_tensor = new_params[k].to(v.device)
            if v.is_floating_point():
                new_state[k] = v + delta_tensor
            else:
                new_state[k] = v + delta_tensor.round().to(v.dtype)
                
        self.global_model.load_state_dict(new_state)
        
        try:
            eval_result = self.evaluator.evaluate(self.global_model, self.eval_dataloader)
        except Exception as e:
            # Atomic rollback: If evaluation crashes, restore the old weights and abort
            self.global_model.load_state_dict(previous_state)
            raise RuntimeError(f"Round failed during evaluation: {e}")

        # 5. Atomic Commit: Everything succeeded, so we checkpoint and advance state
        self.checkpoint_manager.save(target_version, self.global_model.state_dict())
        
        self.current_round = target_round
        self.current_model_version = target_version

        # 6. Result structuring
        round_result = RoundResult(
            round_id=self.current_round,
            model_version=self.current_model_version,
            updates=updates,
            evaluation_result=eval_result,
            decisions=decisions,
        )
        
        self.history.append(round_result)
        return round_result
