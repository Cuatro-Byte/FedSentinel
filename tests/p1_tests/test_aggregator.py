import unittest
from datetime import datetime
import torch

from core.p1_models.model_update import ModelUpdate
from core.federated.aggregator import Aggregator


class TestAggregator(unittest.TestCase):
    def setUp(self):
        self.aggregator = Aggregator()
        self.base_version = "v1"

    def _create_update(self, update_id: str, sample_count: int, params: dict, base_version: str = "v1"):
        return ModelUpdate(
            update_id=update_id,
            run_id="run1",
            round_id=1,
            client_id=f"client_{update_id}",
            model_version=f"{update_id}_v",
            base_model_version=base_version,
            parameters=params,
            sample_count=sample_count,
            local_loss=0.1,
            local_accuracy=0.9,
            training_epochs=1,
            learning_rate=0.01,
            created_at=datetime.now(),
            metadata={}
        )

    def test_aggregator_initialization(self):
        """Test basic initialization."""
        self.assertIsNotNone(self.aggregator)

    def test_single_update_aggregation(self):
        """Test aggregating a single update just returns its parameters cleanly."""
        params = {"layer1": torch.tensor([1.0, 2.0])}
        update = self._create_update("u1", 10, params)
        
        result = self.aggregator.aggregate([update])
        self.assertIn("layer1", result)
        self.assertTrue(torch.allclose(result["layer1"], torch.tensor([1.0, 2.0])))

    def test_standard_fedavg_numerical_correctness(self):
        """Test standard FedAvg with manually calculable tensors."""
        # Update A: weight = 2, params = [1.0, 3.0]
        params_a = {"layer1": torch.tensor([1.0, 3.0])}
        upd_a = self._create_update("uA", 2, params_a)
        
        # Update B: weight = 6, params = [5.0, 7.0]
        params_b = {"layer1": torch.tensor([5.0, 7.0])}
        upd_b = self._create_update("uB", 6, params_b)
        
        result = self.aggregator.aggregate([upd_a, upd_b])
        
        # Expected: [(2*1 + 6*5)/8, (2*3 + 6*7)/8]
        # Expected: [(2 + 30)/8, (6 + 42)/8]
        # Expected: [32/8, 48/8]
        # Expected: [4.0, 6.0]
        expected = torch.tensor([4.0, 6.0])
        
        self.assertTrue(torch.allclose(result["layer1"], expected))

    def test_weighted_aggregation_numerical_correctness(self):
        """Test externally weighted aggregation with manually calculable tensors."""
        params_a = {"layer1": torch.tensor([1.0, 3.0])}
        upd_a = self._create_update("uA", 2, params_a)
        
        params_b = {"layer1": torch.tensor([5.0, 7.0])}
        upd_b = self._create_update("uB", 6, params_b)
        
        # External weights: uA gets 0.5, uB gets 0.0 (excluded)
        # Eff A: 2 * 0.5 = 1.0
        # Eff B: 6 * 0.0 = 0.0
        # Total weight = 1.0
        # Result should be exactly A's parameters
        weights = {"uA": 0.5, "uB": 0.0}
        
        result = self.aggregator.aggregate_weighted([upd_a, upd_b], weights)
        
        expected = torch.tensor([1.0, 3.0])
        self.assertTrue(torch.allclose(result["layer1"], expected))

    def test_incompatible_parameter_keys_rejected(self):
        """Test rejection when updates have different keys."""
        upd_a = self._create_update("uA", 2, {"layer1": torch.tensor([1.0])})
        upd_b = self._create_update("uB", 2, {"layer2": torch.tensor([1.0])})
        
        with self.assertRaises(ValueError):
            self.aggregator.aggregate([upd_a, upd_b])

    def test_incompatible_tensor_shapes_rejected(self):
        """Test rejection when tensor shapes differ."""
        upd_a = self._create_update("uA", 2, {"l1": torch.tensor([1.0, 2.0])})
        upd_b = self._create_update("uB", 2, {"l1": torch.tensor([1.0])})
        
        with self.assertRaises(ValueError):
            self.aggregator.aggregate([upd_a, upd_b])

    def test_empty_updates_rejected(self):
        """Test empty lists are rejected."""
        with self.assertRaises(ValueError):
            self.aggregator.aggregate([])

    def test_invalid_sample_counts_rejected(self):
        """Test negative or zero sample counts are rejected."""
        upd_a = self._create_update("uA", 0, {"l1": torch.tensor([1.0])})
        with self.assertRaises(ValueError):
            self.aggregator.aggregate([upd_a])

    def test_different_base_versions_rejected(self):
        """Test base model version mismatch."""
        upd_a = self._create_update("uA", 2, {"l1": torch.tensor([1.0])}, "v1")
        upd_b = self._create_update("uB", 2, {"l1": torch.tensor([1.0])}, "v2")
        
        with self.assertRaises(ValueError):
            self.aggregator.aggregate([upd_a, upd_b])

    def test_tensor_immutability(self):
        """Test that original tensors are not mutated."""
        t_a = torch.tensor([1.0, 2.0])
        t_a_clone = t_a.clone()
        upd_a = self._create_update("uA", 2, {"l1": t_a})
        
        t_b = torch.tensor([3.0, 4.0])
        t_b_clone = t_b.clone()
        upd_b = self._create_update("uB", 2, {"l1": t_b})
        
        self.aggregator.aggregate([upd_a, upd_b])
        
        self.assertTrue(torch.allclose(t_a, t_a_clone))
        self.assertTrue(torch.allclose(t_b, t_b_clone))

    def test_invalid_external_weights(self):
        """Test invalid weights handling."""
        upd_a = self._create_update("uA", 2, {"l1": torch.tensor([1.0])})
        
        # Missing weight
        with self.assertRaises(ValueError):
            self.aggregator.aggregate_weighted([upd_a], {})
            
        # Negative weight
        with self.assertRaises(ValueError):
            self.aggregator.aggregate_weighted([upd_a], {"uA": -1.0})
            
        # All zero weights
        with self.assertRaises(ValueError):
            self.aggregator.aggregate_weighted([upd_a], {"uA": 0.0})

    def test_nan_values_rejected(self):
        """Test that NaN values in updates are caught and rejected."""
        upd_a = self._create_update("uA", 2, {"l1": torch.tensor([float('nan')])})
        with self.assertRaises(ValueError):
            self.aggregator.aggregate([upd_a])

    def test_exact_zero_weight_behavior(self):
        """Test the explicit zero-weight behavior as requested by verification."""
        upd_a = self._create_update("uA", 10, {"layer1": torch.tensor([1.0])})
        upd_b = self._create_update("uB", 10, {"layer1": torch.tensor([5.0])})
        
        weights = {"uA": 0.0, "uB": 1.0}
        
        result = self.aggregator.aggregate_weighted([upd_a, upd_b], weights)
        
        expected = torch.tensor([5.0])
        self.assertTrue(torch.allclose(result["layer1"], expected))

if __name__ == '__main__':
    unittest.main()
