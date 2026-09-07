import unittest
import torch
from core.models.model import SimpleCNN

class TestSimpleCNN(unittest.TestCase):
    def setUp(self):
        # Set a manual seed for reproducible random initialization if needed
        torch.manual_seed(42)
        self.model = SimpleCNN(in_channels=1, num_classes=10)

    def test_model_instantiation(self):
        """Test that the model can be instantiated."""
        self.assertIsInstance(self.model, SimpleCNN)
        self.assertTrue(hasattr(self.model, 'conv1'))
        self.assertTrue(hasattr(self.model, 'fc2'))

    def test_forward_pass_and_shape(self):
        """Test the forward pass and verify the output shape."""
        batch_size = 4
        # Dummy input for 28x28 grayscale images
        x = torch.randn(batch_size, 1, 28, 28)
        output = self.model(x)
        
        # Output should be (batch_size, num_classes)
        self.assertEqual(output.shape, (batch_size, 10))

    def test_get_and_set_weights(self):
        """Test extracting and loading the state_dict."""
        # Extract weights from the original model
        weights = self.model.get_weights()
        
        # Ensure it returns a dictionary with tensor values
        self.assertIsInstance(weights, dict)
        self.assertTrue(len(weights) > 0)
        
        # Instantiate a second, different model
        model2 = SimpleCNN(in_channels=1, num_classes=10)
        
        # Modify model2 weights slightly to ensure they are different initially
        with torch.no_grad():
            for param in model2.parameters():
                param.add_(1.0)
                
        # Generate dummy input
        x = torch.randn(2, 1, 28, 28)
        
        # Assert they produce different outputs initially
        out1 = self.model(x)
        out2 = model2(x)
        self.assertFalse(torch.allclose(out1, out2))
        
        # Load the original weights into the second model
        model2.set_weights(weights)
        
        # Assert they now produce the exact same output
        out2_after_load = model2(x)
        self.assertTrue(torch.allclose(out1, out2_after_load))

    def test_invalid_input_shape(self):
        """Test behavior on invalid input shape."""
        # Provide incorrect shape (e.g., missing channel dimension or wrong spatial size)
        # Using a 10x10 image instead of expected 28x28 (which fails at the linear layer)
        x_wrong_size = torch.randn(1, 1, 10, 10)
        with self.assertRaises(RuntimeError):
            self.model(x_wrong_size)

if __name__ == '__main__':
    unittest.main()
