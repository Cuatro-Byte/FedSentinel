import unittest
import torch

from core.federated.checkpoint import CheckpointManager


class TestCheckpointManager(unittest.TestCase):
    def setUp(self):
        self.manager = CheckpointManager()
        self.state_v0 = {
            "layer1.weight": torch.tensor([1.0, 2.0]),
            "layer1.bias": torch.tensor([0.1])
        }
        self.state_v1 = {
            "layer1.weight": torch.tensor([3.0, 4.0]),
            "layer1.bias": torch.tensor([0.5])
        }

    def test_initialization(self):
        """Test the checkpoint manager initializes empty cleanly."""
        self.assertIsNotNone(self.manager)
        self.assertEqual(len(self.manager.list_versions()), 0)

    def test_save_and_load_checkpoint(self):
        """Test saving and then loading a valid checkpoint."""
        self.manager.save("model_v0", self.state_v0)
        
        self.assertTrue(self.manager.exists("model_v0"))
        
        loaded = self.manager.load("model_v0")
        self.assertIn("layer1.weight", loaded)
        self.assertTrue(torch.allclose(loaded["layer1.weight"], self.state_v0["layer1.weight"]))

    def test_original_state_mutation_does_not_affect_checkpoint(self):
        """Test that changing the caller's tensor after saving does not corrupt history."""
        self.manager.save("model_v0", self.state_v0)
        
        # Mutate original tensor explicitly
        self.state_v0["layer1.weight"][0] = 999.0
        
        loaded = self.manager.load("model_v0")
        
        # The checkpoint should retain the original 1.0 value
        self.assertEqual(loaded["layer1.weight"][0].item(), 1.0)

    def test_loaded_state_mutation_does_not_affect_checkpoint(self):
        """Test that modifying a retrieved checkpoint does not corrupt the internal store."""
        self.manager.save("model_v0", self.state_v0)
        
        loaded1 = self.manager.load("model_v0")
        loaded1["layer1.weight"][0] = 999.0
        
        loaded2 = self.manager.load("model_v0")
        
        # The second load should still yield the original 1.0 value
        self.assertEqual(loaded2["layer1.weight"][0].item(), 1.0)

    def test_multiple_versions_and_listing(self):
        """Test storing multiple versions and listing them deterministically."""
        self.manager.save("model_v0", self.state_v0)
        self.manager.save("model_v1", self.state_v1)
        
        versions = self.manager.list_versions()
        self.assertEqual(versions, ["model_v0", "model_v1"])
        
        # Verify isolation
        v0 = self.manager.load("model_v0")
        v1 = self.manager.load("model_v1")
        self.assertFalse(torch.allclose(v0["layer1.weight"], v1["layer1.weight"]))

    def test_duplicate_version_rejected(self):
        """Test that history cannot be silently overwritten."""
        self.manager.save("model_v0", self.state_v0)
        
        with self.assertRaises(ValueError):
            self.manager.save("model_v0", self.state_v1)

    def test_missing_checkpoint_loading(self):
        """Test loading a missing checkpoint raises KeyError."""
        with self.assertRaises(KeyError):
            self.manager.load("non_existent_version")

    def test_invalid_save_inputs(self):
        """Test rejection of empty strings and invalid state dictionaries."""
        with self.assertRaises(ValueError):
            self.manager.save("", self.state_v0)
            
        with self.assertRaises(ValueError):
            self.manager.save(None, self.state_v0) # type: ignore
            
        with self.assertRaises(ValueError):
            self.manager.save("v1", {})
            
        with self.assertRaises(ValueError):
            self.manager.save("v1", None) # type: ignore

    def test_non_tensor_parameter_rejected(self):
        """Test that all parameters must be valid tensors."""
        bad_state = {"layer1": [1.0, 2.0]} # List instead of tensor
        with self.assertRaises(ValueError):
            self.manager.save("model_v0", bad_state) # type: ignore

    def test_nan_inf_tensors_rejected(self):
        """Test that corrupted tensors are immediately rejected."""
        nan_state = {"l1": torch.tensor([float('nan')])}
        with self.assertRaises(ValueError):
            self.manager.save("v1", nan_state)
            
        inf_state = {"l1": torch.tensor([float('inf')])}
        with self.assertRaises(ValueError):
            self.manager.save("v2", inf_state)


if __name__ == '__main__':
    unittest.main()
