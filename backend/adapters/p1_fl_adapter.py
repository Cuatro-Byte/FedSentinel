import copy
from datetime import datetime
from typing import Dict, List, Any
import torch

from backend.adapters.interfaces import FLCoreInterface
from core.models.model_update import ModelUpdate as P4ModelUpdate
from core.models.enums import ResponseAction as P4ResponseAction, ThreatLevel as P4ThreatLevel

# P1 components
from core.federated.client_manager import ClientManager
from core.federated.aggregator import Aggregator
from core.federated.evaluator import Evaluator
from core.federated.checkpoint import CheckpointManager
from core.p1_models.model_update import ModelUpdate as P1ModelUpdate
from core.p1_models.detection_result import DetectionResult as P1DetectionResult
from core.p1_models.detection_result import ResponseAction as P1ResponseAction
from core.p1_models.detection_result import ThreatLevel as P1ThreatLevel

class P1FLAdapter(FLCoreInterface):
    def __init__(
        self,
        global_model: torch.nn.Module,
        client_manager: ClientManager,
        aggregator: Aggregator,
        evaluator: Evaluator,
        checkpoint_manager: CheckpointManager,
        eval_dataloader=None,
    ):
        self.global_model = global_model
        self.client_manager = client_manager
        self.aggregator = aggregator
        self.evaluator = evaluator
        self.checkpoint_manager = checkpoint_manager
        self.eval_dataloader = eval_dataloader

    def _convert_update_p1_to_p4(self, p1_update: P1ModelUpdate) -> P4ModelUpdate:
        return P4ModelUpdate(
            update_id=p1_update.update_id,
            run_id=p1_update.run_id,
            round_id=p1_update.round_id,
            client_id=p1_update.client_id,
            model_version=p1_update.model_version,
            base_model_version=p1_update.base_model_version,
            parameters=p1_update.parameters,
            sample_count=p1_update.sample_count,
            local_loss=p1_update.local_loss,
            local_accuracy=p1_update.local_accuracy,
            training_epochs=p1_update.training_epochs,
            learning_rate=p1_update.learning_rate,
            created_at=p1_update.created_at,
            metadata=p1_update.metadata,
        )

    def _convert_update_p4_to_p1(self, p4_update: P4ModelUpdate) -> P1ModelUpdate:
        return P1ModelUpdate(
            update_id=p4_update.update_id,
            run_id=p4_update.run_id,
            round_id=p4_update.round_id,
            client_id=p4_update.client_id,
            model_version=p4_update.model_version,
            base_model_version=p4_update.base_model_version,
            parameters=p4_update.parameters,
            sample_count=p4_update.sample_count,
            local_loss=p4_update.local_loss,
            local_accuracy=p4_update.local_accuracy,
            training_epochs=p4_update.training_epochs,
            learning_rate=p4_update.learning_rate,
            created_at=p4_update.created_at,
            metadata=p4_update.metadata,
        )

    def _convert_action_p4_to_p1(self, action: P4ResponseAction) -> P1ResponseAction:
        if action == P4ResponseAction.ACCEPT:
            return P1ResponseAction.ACCEPT
        elif action == P4ResponseAction.DOWN_WEIGHT:
            return P1ResponseAction.DOWN_WEIGHT
        elif action == P4ResponseAction.QUARANTINE:
            return P1ResponseAction.QUARANTINE
        raise ValueError(f"Unknown action {action}")

    def provision_clients(
        self,
        client_count: int,
        seed: int = 42,
        local_epochs: int = 2,
        batch_size: int = 16,
    ) -> None:
        """Provision exactly client_count clients with partitioned real data."""
        from torch.utils.data import DataLoader
        from core.federated.client import SimulatedClient
        from core.federated.trainer import LocalTrainer
        from backend.services.dataset_loader import get_default_datasets, partition_for_clients

        from core.validation.server_validator import ServerValidationGate

        train_ds, val_ds, eval_ds = get_default_datasets(seed=seed)
        partitions = partition_for_clients(train_ds, client_count=client_count, seed=seed)

        self.client_manager = ClientManager()
        for client_id, client_ds in partitions.items():
            loader = DataLoader(client_ds, batch_size=batch_size, shuffle=True)
            trainer = LocalTrainer(epochs=local_epochs)
            self.client_manager.register_client(
                SimulatedClient(client_id=client_id, dataloader=loader, trainer=trainer)
            )

        self.eval_dataloader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)
        self.server_validator = ServerValidationGate(val_dataset=val_ds, batch_size=batch_size)

    def initialize_global_model(self, config: dict) -> str:
        import uuid
        client_count = config.get("client_count")
        if client_count is not None and client_count > 0:
            seed = config.get("seed", 42)
            epochs = config.get("local_epochs", 2)
            self.provision_clients(client_count=client_count, seed=seed, local_epochs=epochs)

        version = config.get("initial_model_version", f"model_v0_{uuid.uuid4().hex[:8]}")
        self.checkpoint_manager.save(version, self.global_model.state_dict())
        return version

    def train_clients(self, run_id: str, round_id: int,
                      global_model_version: str,
                      client_ids: list[str],
                      config: dict) -> list[P4ModelUpdate]:
        
        state_dict = self.checkpoint_manager.load(global_model_version)
        self.global_model.load_state_dict(state_dict)
        
        clients = []
        for cid in client_ids:
            client = self.client_manager.get_client(cid)
            if client:
                clients.append(client)
        
        updates = []
        for client in clients:
            p1_update = client.train(
                global_model=self.global_model,
                run_id=run_id,
                round_id=round_id,
                base_model_version=global_model_version,
                model_version=f"{global_model_version}_{client.client_id}"
            )
            updates.append(self._convert_update_p1_to_p4(p1_update))
        
        return updates

    def _convert_detection_p4_to_p1(self, p4_det) -> P1DetectionResult:
        p1_action = self._convert_action_p4_to_p1(p4_det.action)
        if p4_det.threat_level == P4ThreatLevel.SAFE:
            p1_threat = P1ThreatLevel.SAFE
        elif p4_det.threat_level == P4ThreatLevel.SUSPICIOUS:
            p1_threat = P1ThreatLevel.SUSPICIOUS
        else:
            p1_threat = P1ThreatLevel.MALICIOUS
            
        return P1DetectionResult(
            update_id=p4_det.update_id,
            client_id=p4_det.client_id,
            round_id=p4_det.round_id,
            threat_score=p4_det.threat_score,
            threat_level=p1_threat,
            action=p1_action,
            feature_summary=p4_det.feature_summary,
            anomaly_score=p4_det.anomaly_score,
            similarity_score=p4_det.similarity_score,
            reputation_score=p4_det.reputation_score,
            explanation_codes=p4_det.explanation_codes,
            detector_version=p4_det.detector_version,
            created_at=p4_det.created_at
        )

    def aggregate(self, updates: list[P4ModelUpdate],
                  detections: list[Any],
                  current_model_version: str) -> str:
        
        p1_updates = [self._convert_update_p4_to_p1(u) for u in updates]
        p1_decisions = [self._convert_detection_p4_to_p1(d) for d in detections]

        # Filter out quarantined updates so corrupted/empty tensors do not enter aggregator
        non_quarantined = [
            (u, d) for u, d in zip(p1_updates, p1_decisions)
            if d.action != P1ResponseAction.QUARANTINE
        ]

        base_state = self.checkpoint_manager.load(current_model_version)
        round_id = updates[0].round_id if updates else 0
        run_id = updates[0].run_id if updates else "unknown_run"
        new_version = f"model_{run_id}_v{round_id}"

        if non_quarantined:
            filtered_updates, filtered_decisions = zip(*non_quarantined)
            new_params = self.aggregator.aggregate_with_decisions(
                list(filtered_updates), list(filtered_decisions)
            )
            new_state = {}
            for k, v in base_state.items():
                delta_tensor = new_params[k].to(v.device)
                if v.is_floating_point():
                    new_state[k] = v + delta_tensor
                else:
                    new_state[k] = v + delta_tensor.round().to(v.dtype)
        else:
            # If all updates are quarantined, preserve current model state
            new_state = {k: v.clone() for k, v in base_state.items()}

        self.global_model.load_state_dict(new_state)
        self.checkpoint_manager.save(new_version, new_state)

        return new_version

    def validate_candidate(self, model_version: str) -> tuple[dict, bool]:
        state_dict = self.checkpoint_manager.load(model_version)
        self.global_model.load_state_dict(state_dict)
        if not hasattr(self, 'server_validator'):
            return {"val_loss": 0.0, "val_acc": 1.0, "loss_delta": 0.0, "acc_delta": 0.0}, False
        return self.server_validator.evaluate_checkpoint(self.global_model)

    def evaluate(self, model_version: str) -> dict:
        state_dict = self.checkpoint_manager.load(model_version)
        self.global_model.load_state_dict(state_dict)
        
        if self.eval_dataloader is None:
            return {"accuracy": 0.0, "loss": 0.0}
            
        result = self.evaluator.evaluate(self.global_model, self.eval_dataloader)
        return {
            "accuracy": result.accuracy,
            "loss": result.loss
        }

    def re_aggregate(self, updates: list[P4ModelUpdate],
                     excluded_client_ids: list[str],
                     previous_model_version: str) -> str:
                     
        p1_updates = [self._convert_update_p4_to_p1(u) for u in updates]
        
        filtered_updates = [u for u in p1_updates if u.client_id not in excluded_client_ids]
        
        if not filtered_updates:
            return previous_model_version
            
        new_params = self.aggregator.aggregate(filtered_updates)
        
        base_state = self.checkpoint_manager.load(previous_model_version)
        new_state = {}
        for k, v in base_state.items():
            delta_tensor = new_params[k].to(v.device)
            if v.is_floating_point():
                new_state[k] = v + delta_tensor
            else:
                new_state[k] = v + delta_tensor.round().to(v.dtype)
                
        self.global_model.load_state_dict(new_state)
        
        round_id = updates[0].round_id if updates else 0
        run_id = updates[0].run_id if updates else "unknown_run"
        new_version = f"model_{run_id}_v{round_id}_recovered"
        self.checkpoint_manager.save(new_version, new_state)
        
        return new_version
