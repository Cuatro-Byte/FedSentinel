import unittest
import torch
from torch.utils.data import TensorDataset, DataLoader
from core.p1_models.model import SimpleCNN
from core.federated.evaluator import Evaluator, EvaluationResult


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = SimpleCNN(in_channels=1, num_classes=10)
        
        # Tiny synthetic dataset
        self.x_data = torch.randn(8, 1, 28, 28)
        self.y_data = torch.randint(0, 10, (8,))
        self.dataset = TensorDataset(self.x_data, self.y_data)
        self.dataloader = DataLoader(self.dataset, batch_size=4)
        
        self.evaluator = Evaluator(device="cpu")

    def test_evaluator_initialization(self):
        """Test valid evaluator initialization."""
        self.assertEqual(self.evaluator.device.type, "cpu")
        self.assertIsNotNone(self.evaluator.criterion)

    def test_successful_evaluation(self):
        """Test that a standard evaluation succeeds and returns valid metrics."""
        result = self.evaluator.evaluate(self.model, self.dataloader)
        
        self.assertIsInstance(result, EvaluationResult)
        
        # Loss should be finite and positive
        self.assertIsInstance(result.loss, float)
        self.assertTrue(result.loss > 0)
        
        # Accuracy should be in [0, 1]
        self.assertIsInstance(result.accuracy, float)
        self.assertTrue(0.0 <= result.accuracy <= 1.0)
        
        # Sample count should exactly match
        self.assertEqual(result.sample_count, 8)

    def test_model_immutability(self):
        """Test that model weights do not change during evaluation."""
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        self.evaluator.evaluate(self.model, self.dataloader)
        
        for k, v in self.model.state_dict().items():
            self.assertTrue(torch.allclose(v, original_state[k]))

    def test_no_gradients_accumulated(self):
        """Test that evaluation does not calculate or accumulate gradients."""
        # Ensure gradients are zeroed out initially
        self.model.zero_grad()
        
        self.evaluator.evaluate(self.model, self.dataloader)
        
        # Check that no parameter has a gradient
        for param in self.model.parameters():
            self.assertIsNone(param.grad)

    def test_mode_restoration(self):
        """Test that the evaluator restores the original training/eval mode."""
        # Force model into train mode initially
        self.model.train()
        self.evaluator.evaluate(self.model, self.dataloader)
        self.assertTrue(self.model.training, "Model should be restored to train mode.")
        
        # Force model into eval mode initially
        self.model.eval()
        self.evaluator.evaluate(self.model, self.dataloader)
        self.assertFalse(self.model.training, "Model should be restored to eval mode.")

    def test_invalid_evaluation_inputs(self):
        """Test evaluator validates model, dataloader, and empty datasets."""
        with self.assertRaises(ValueError):
            self.evaluator.evaluate(None, self.dataloader) # type: ignore
            
        with self.assertRaises(ValueError):
            self.evaluator.evaluate(self.model, None) # type: ignore
            
        # Empty dataloader
        empty_dataset = TensorDataset(torch.empty(0, 1), torch.empty(0))
        empty_dataloader = DataLoader(empty_dataset)
        with self.assertRaises(ValueError):
            self.evaluator.evaluate(self.model, empty_dataloader)

    def test_malformed_batch(self):
        """Test evaluator rejects malformed batches gracefully."""
        class BadDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 4
            def __getitem__(self, idx):
                return torch.randn(1, 28, 28) # Missing target
                
        bad_loader = DataLoader(BadDataset(), batch_size=2)
        
        with self.assertRaises(ValueError):
            self.evaluator.evaluate(self.model, bad_loader)

    def test_consistent_evaluation(self):
        """Test that repeated evaluation on the same data/model yields identical results."""
        result1 = self.evaluator.evaluate(self.model, self.dataloader)
        result2 = self.evaluator.evaluate(self.model, self.dataloader)
        
        self.assertEqual(result1.loss, result2.loss)
        self.assertEqual(result1.accuracy, result2.accuracy)
        self.assertEqual(result1.sample_count, result2.sample_count)


if __name__ == '__main__':
    unittest.main()
