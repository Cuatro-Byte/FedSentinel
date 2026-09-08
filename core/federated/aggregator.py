import torch
from typing import List, Dict, Any, Optional

from core.p1_models.model_update import ModelUpdate
from core.p1_models.detection_result import DetectionResult, ResponseAction


class Aggregator:
    """
    Infrastructure component responsible for combining federated model updates 
    into a new global model state.
    
    It supports standard Federated Averaging (FedAvg) and trust-aware weighted 
    aggregation. It strictly performs mathematical aggregation and does not contain 
    threat detection, scoring, or maliciousness logic.
    """

    def aggregate(self, updates: List[ModelUpdate]) -> Dict[str, torch.Tensor]:
        """
        Perform standard Federated Averaging (FedAvg).
        Each update contributes proportionally to its sample_count.
        
        Args:
            updates: A list of valid ModelUpdate objects to aggregate.
            
        Returns:
            A dictionary mapping parameter names to the aggregated torch Tensors.
            
        Raises:
            ValueError: If the updates list is empty, malformed, or incompatible.
        """
        return self._aggregate_impl(updates, external_weights=None)

    def aggregate_weighted(
        self, 
        updates: List[ModelUpdate], 
        external_weights: Dict[str, float]
    ) -> Dict[str, torch.Tensor]:
        """
        Perform trust-aware weighted aggregation based on externally supplied weights.
        The effective weight is: update.sample_count * external_weights[update.update_id].
        
        Args:
            updates: A list of valid ModelUpdate objects to aggregate.
            external_weights: A mapping from update_id to a non-negative float weight.
            
        Returns:
            A dictionary mapping parameter names to the aggregated torch Tensors.
            
        Raises:
            ValueError: If the updates or weights are invalid, missing, or incompatible.
        """
        if external_weights is None or not isinstance(external_weights, dict):
            raise ValueError("external_weights must be a valid dictionary.")
        return self._aggregate_impl(updates, external_weights=external_weights)

    def aggregate_with_decisions(
        self,
        updates: List[ModelUpdate],
        decisions: List[DetectionResult],
        down_weight_factor: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform trust-aware aggregation using canonical DetectionResult decisions.

        Each update's aggregation weight is determined by its action:
          ACCEPT      → 1.0  (full sample-count weight)
          DOWN_WEIGHT → down_weight_factor  (configurable reduced weight, default 0.5)
          QUARANTINE  → 0.0  (completely excluded)

        Effective weight = update.sample_count × action_weight.

        Args:
            updates: A list of valid ModelUpdate objects to aggregate.
            decisions: A list of DetectionResult objects, one per update.
            down_weight_factor: Scalar in (0, 1) applied to DOWN_WEIGHT updates.

        Returns:
            A dictionary mapping parameter names to aggregated torch Tensors.

        Raises:
            ValueError: If decisions are missing, duplicate, mismatched, or all quarantined.
        """
        if not isinstance(down_weight_factor, (int, float)) or not (0.0 < down_weight_factor < 1.0):
            raise ValueError(f"down_weight_factor must be a float in (0, 1), got {down_weight_factor!r}.")

        update_ids = [u.update_id for u in updates]

        # Validate decision list
        if not decisions:
            raise ValueError("aggregate_with_decisions: decisions list is empty.")

        decision_map: Dict[str, DetectionResult] = {}
        for dr in decisions:
            if not isinstance(dr, DetectionResult):
                raise ValueError(f"aggregate_with_decisions: item in decisions is not a DetectionResult.")
            if dr.update_id in decision_map:
                raise ValueError(f"aggregate_with_decisions: duplicate DetectionResult for update_id '{dr.update_id}'.")
            decision_map[dr.update_id] = dr

        # Every update must have exactly one decision
        for uid in update_ids:
            if uid not in decision_map:
                raise ValueError(f"aggregate_with_decisions: missing DetectionResult for update_id '{uid}'.")

        # Extra decisions for updates not in this aggregation set are rejected
        for uid in decision_map:
            if uid not in update_ids:
                raise ValueError(f"aggregate_with_decisions: DetectionResult references unknown update_id '{uid}'.")

        # Build action → numeric weight mapping
        action_weight_map = {
            ResponseAction.ACCEPT: 1.0,
            ResponseAction.DOWN_WEIGHT: down_weight_factor,
            ResponseAction.QUARANTINE: 0.0,
        }

        external_weights: Dict[str, float] = {
            uid: action_weight_map[decision_map[uid].action]
            for uid in update_ids
        }

        return self._aggregate_impl(updates, external_weights=external_weights)

    def _aggregate_impl(
        self, 
        updates: List[ModelUpdate], 
        external_weights: Dict[str, float] | None
    ) -> Dict[str, torch.Tensor]:
        """Internal implementation handling both standard and weighted aggregation safely."""
        self._validate_updates(updates)

        # 1. Calculate effective weights
        total_weight = 0.0
        effective_weights = []

        for update in updates:
            base_weight = float(update.sample_count)
            if external_weights is not None:
                if update.update_id not in external_weights:
                    raise ValueError(f"Missing external weight for update {update.update_id}.")
                ext_w = external_weights[update.update_id]
                if ext_w < 0:
                    raise ValueError(f"External weight for update {update.update_id} cannot be negative.")
                weight = base_weight * ext_w
            else:
                weight = base_weight
                
            effective_weights.append(weight)
            total_weight += weight

        if total_weight <= 0.0:
            raise ValueError("Total effective weight is zero or negative. Cannot aggregate.")

        # 2. Extract reference structures from the first update to validate compatibility
        reference_params = updates[0].parameters
        reference_keys = set(reference_params.keys())

        # Initialize the aggregated state dict
        aggregated_state: Dict[str, torch.Tensor] = {}
        for key in reference_keys:
            # Clone and initialize to zero to prevent mutating any source tensor
            aggregated_state[key] = torch.zeros_like(reference_params[key], dtype=torch.float32)

        # 3. Accumulate weighted parameters
        for update, eff_weight in zip(updates, effective_weights):
            if eff_weight == 0.0:
                continue # Skip computationally if weight is zero
                
            params = update.parameters
            
            # Validate structural compatibility
            if set(params.keys()) != reference_keys:
                raise ValueError(f"Update {update.update_id} has incompatible parameter keys.")

            # Calculate normalized contribution
            normalized_weight = eff_weight / total_weight

            for key in reference_keys:
                tensor = params[key]
                if tensor.shape != reference_params[key].shape:
                    raise ValueError(f"Update {update.update_id} tensor {key} shape mismatch.")
                
                # Check for NaNs or Infs
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    raise ValueError(f"Update {update.update_id} tensor {key} contains NaN/Inf values.")

                # Accumulate safely
                aggregated_state[key] += tensor.to(torch.float32) * normalized_weight

        # Return the new state dict cleanly cast back to original dtypes if necessary
        for key in reference_keys:
            aggregated_state[key] = aggregated_state[key].to(reference_params[key].dtype)

        return aggregated_state

    def _validate_updates(self, updates: List[ModelUpdate]) -> None:
        """Validate the structural and contextual integrity of the update collection."""
        if not updates:
            raise ValueError("Cannot aggregate an empty list of updates.")

        reference_base_version = None

        for idx, update in enumerate(updates):
            if not isinstance(update, ModelUpdate):
                raise ValueError(f"Item at index {idx} is not a ModelUpdate instance.")
            
            if update.sample_count <= 0:
                raise ValueError(f"Update {update.update_id} has invalid sample_count {update.sample_count}.")
                
            if not update.parameters:
                raise ValueError(f"Update {update.update_id} has empty parameters.")

            if reference_base_version is None:
                reference_base_version = update.base_model_version
            elif update.base_model_version != reference_base_version:
                raise ValueError(
                    f"Incompatible base model versions: expected {reference_base_version}, "
                    f"but update {update.update_id} has {update.base_model_version}."
                )
