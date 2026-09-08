import unittest
import torch
from torch.utils.data import TensorDataset, DataLoader
from core.p1_models.model import SimpleCNN
from core.p1_models.model_update import ModelUpdate
from core.federated.trainer import LocalTrainer
from core.federated.client import SimulatedClient

class TestSimulatedClient(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.global_model = SimpleCNN(in_channels=1, num_classes=10)
        
        # Tiny synthetic dataset
        self.x_data = torch.randn(4, 1, 28, 28)
        self.y_data = torch.randint(0, 10, (4,))
        self.dataset = TensorDataset(self.x_data, self.y_data)
        self.dataloader = DataLoader(self.dataset, batch_size=2)
        
        self.trainer = LocalTrainer(learning_rate=0.01, epochs=1, device="cpu")
        self.client = SimulatedClient("client_123", self.dataloader, self.trainer)

    def test_client_initialization(self):
        """Test valid client initialization and invalid configurations."""
        self.assertEqual(self.client.client_id, "client_123")
        self.assertIs(self.client.dataloader, self.dataloader)
        self.assertIs(self.client.trainer, self.trainer)
        
        with self.assertRaises(ValueError):
            SimulatedClient("", self.dataloader, self.trainer)
            
        with self.assertRaises(ValueError):
            SimulatedClient("client_1", None, self.trainer)
            
        with self.assertRaises(ValueError):
            SimulatedClient("client_1", self.dataloader, None)

    def test_train_and_model_update_creation(self):
        """Test that the client can train and return a valid ModelUpdate."""
        
        # Capture original global model state to ensure it doesn't mutate
        original_state = {k: v.clone() for k, v in self.global_model.state_dict().items()}
        
        update = self.client.train(
            global_model=self.global_model,
            run_id="run_abc",
            round_id=5,
            base_model_version="v_base",
            model_version="v_local"
        )
        
        # Verify the global model was not mutated
        for k, v in self.global_model.state_dict().items():
            self.assertTrue(torch.allclose(v, original_state[k]))
        
        # Verify returned type
        self.assertIsInstance(update, ModelUpdate)
        
        # Verify ModelUpdate fields
        self.assertIsInstance(update.update_id, str)
        self.assertTrue(len(update.update_id) > 10) # Valid UUID
        
        self.assertEqual(update.client_id, "client_123")
        self.assertEqual(update.run_id, "run_abc")
        self.assertEqual(update.round_id, 5)
        self.assertEqual(update.base_model_version, "v_base")
        self.assertEqual(update.model_version, "v_local")
        
        # Verify metrics
        self.assertEqual(update.sample_count, 4)
        self.assertIsInstance(update.local_loss, float)
        self.assertIsInstance(update.local_accuracy, float)
        self.assertEqual(update.training_epochs, 1)
        self.assertEqual(update.learning_rate, 0.01)
        
        # Verify parameters contain actual trained state
        self.assertIsInstance(update.parameters, dict)
        self.assertTrue(len(update.parameters) > 0)
        
        # Verify metadata does not contain maliciousness flags
        self.assertNotIn("is_malicious", update.metadata)
        self.assertNotIn("threat_score", update.metadata)

    def test_unique_update_ids(self):
        """Test that calling train multiple times generates unique update IDs."""
        update1 = self.client.train(self.global_model, "run_1", 1, "v1", "v1_loc")
        update2 = self.client.train(self.global_model, "run_1", 2, "v2", "v2_loc")
        
        self.assertNotEqual(update1.update_id, update2.update_id)

    def test_independent_clients(self):
        """Test that two independent clients can train independently."""
        client2 = SimulatedClient("client_456", self.dataloader, self.trainer)
        
        update1 = self.client.train(self.global_model, "r1", 1, "v0", "v1")
        update2 = client2.train(self.global_model, "r1", 1, "v0", "v2")
        
        self.assertNotEqual(update1.client_id, update2.client_id)
        self.assertNotEqual(update1.update_id, update2.update_id)
        
    def test_invalid_train_arguments(self):
        """Test validation on train arguments."""
        with self.assertRaises(ValueError):
            self.client.train(None, "r1", 1, "v0", "v1")
            
        with self.assertRaises(ValueError):
            self.client.train(self.global_model, "", 1, "v0", "v1")
            
        with self.assertRaises(ValueError):
            self.client.train(self.global_model, "r1", -1, "v0", "v1")

if __name__ == '__main__':
    unittest.main()
