import unittest
import torch
from torch.utils.data import TensorDataset, DataLoader
from core.models.model import SimpleCNN
from core.federated.trainer import LocalTrainer, TrainingResult

class TestLocalTrainer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = SimpleCNN(in_channels=1, num_classes=10)
        
        # Create a tiny synthetic dataset (e.g., 8 samples of 28x28)
        self.x_data = torch.randn(8, 1, 28, 28)
        # Target classes between 0 and 9
        self.y_data = torch.randint(0, 10, (8,))
        
        self.dataset = TensorDataset(self.x_data, self.y_data)
        self.dataloader = DataLoader(self.dataset, batch_size=4, shuffle=True)

    def test_trainer_initialization(self):
        """Test that the trainer initializes with valid params and rejects invalid ones."""
        trainer = LocalTrainer(learning_rate=0.01, epochs=2, device="cpu")
        self.assertEqual(trainer.learning_rate, 0.01)
        self.assertEqual(trainer.epochs, 2)
        
        with self.assertRaises(ValueError):
            LocalTrainer(learning_rate=-0.1)
            
        with self.assertRaises(ValueError):
            LocalTrainer(epochs=0)

    def test_training_execution(self):
        """Test that training runs, changes parameters, and returns valid TrainingResult."""
        trainer = LocalTrainer(learning_rate=0.05, epochs=1, device="cpu")
        
        # Capture parameters before training
        initial_params = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        result = trainer.train(self.model, self.dataloader)
        
        # Verify the returned object type
        self.assertIsInstance(result, TrainingResult)
        
        # Check metrics
        self.assertIsInstance(result.loss, float)
        self.assertTrue(result.loss > 0)
        
        self.assertIsInstance(result.accuracy, float)
        self.assertTrue(0.0 <= result.accuracy <= 1.0)
        
        self.assertEqual(result.sample_count, 8)
        self.assertEqual(result.epochs_completed, 1)
        
        # Verify that parameters actually changed
        changed = False
        for k, initial_v in initial_params.items():
            current_v = result.model_state[k]
            if not torch.allclose(initial_v, current_v):
                changed = True
                break
                
        self.assertTrue(changed, "Model parameters did not change after training.")

    def test_configurable_epochs(self):
        """Test that configuring multiple epochs runs successfully."""
        trainer = LocalTrainer(learning_rate=0.01, epochs=3, device="cpu")
        result = trainer.train(self.model, self.dataloader)
        self.assertEqual(result.epochs_completed, 3)
        self.assertEqual(result.sample_count, 8)
        
    def test_invalid_training_inputs(self):
        """Test that the train method validates inputs properly."""
        trainer = LocalTrainer()
        
        # Test missing model
        with self.assertRaises(ValueError):
            trainer.train(None, self.dataloader)
            
        # Test empty dataset
        empty_dataset = TensorDataset(torch.empty(0, 1, 28, 28), torch.empty(0, dtype=torch.long))
        empty_dataloader = DataLoader(empty_dataset)
        
        with self.assertRaises(ValueError):
            trainer.train(self.model, empty_dataloader)
            
    def test_malformed_batch(self):
        """Test behavior when dataloader yields malformed batches."""
        # Yields just one tensor instead of (data, target)
        class BadDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 4
            def __getitem__(self, idx):
                return torch.randn(1, 28, 28)
                
        bad_loader = DataLoader(BadDataset(), batch_size=2)
        trainer = LocalTrainer()
        
        with self.assertRaises(ValueError):
            trainer.train(self.model, bad_loader)

if __name__ == '__main__':
    unittest.main()
