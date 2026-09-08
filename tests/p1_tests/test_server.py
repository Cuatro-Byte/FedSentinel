import unittest
import torch
from torch.utils.data import TensorDataset, DataLoader

from core.p1_models.model import SimpleCNN
from core.federated.trainer import LocalTrainer
from core.federated.client import SimulatedClient
from core.federated.client_manager import ClientManager
from core.federated.aggregator import Aggregator
from core.federated.evaluator import Evaluator
from core.federated.server import Server, RoundResult


class TestServer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        
        self.global_model = SimpleCNN(in_channels=1, num_classes=10)
        self.client_manager = ClientManager()
        self.aggregator = Aggregator()
        self.evaluator = Evaluator(device="cpu")
        
        # Tiny synthetic eval dataset
        eval_ds = TensorDataset(torch.randn(4, 1, 28, 28), torch.randint(0, 10, (4,)))
        self.eval_dataloader = DataLoader(eval_ds, batch_size=2)
        
        # Tiny synthetic client datasets and trainer
        self.trainer = LocalTrainer(learning_rate=0.01, epochs=1, device="cpu")
        for i in range(3):
            ds = TensorDataset(torch.randn(2, 1, 28, 28), torch.randint(0, 10, (2,)))
            dl = DataLoader(ds, batch_size=2)
            client = SimulatedClient(f"client_{i}", dl, self.trainer)
            self.client_manager.register_client(client)
            
        self.server = Server(
            global_model=self.global_model,
            client_manager=self.client_manager,
            aggregator=self.aggregator,
            evaluator=self.evaluator,
            run_id="run_test",
            initial_model_version="model_v0",
            eval_dataloader=self.eval_dataloader
        )

    def test_server_initialization_and_validation(self):
        """Test valid initialization and rejection of invalid configurations."""
        self.assertEqual(self.server.current_round, 0)
        self.assertEqual(self.server.current_model_version, "model_v0")
        self.assertEqual(self.server.run_id, "run_test")
        
        with self.assertRaises(ValueError):
            Server(None, self.client_manager, self.aggregator, self.evaluator, "run1") # type: ignore
        with self.assertRaises(ValueError):
            Server(self.global_model, self.client_manager, self.aggregator, self.evaluator, "")

    def test_execute_round_success(self):
        """Test a complete normal FL round execution."""
        original_state = {k: v.clone() for k, v in self.global_model.state_dict().items()}
        
        # Execute round with 2 clients
        result = self.server.execute_round(num_clients=2, seed=123)
        
        self.assertIsInstance(result, RoundResult)
        
        # Round numbering
        self.assertEqual(result.round_id, 1)
        self.assertEqual(self.server.current_round, 1)
        
        # Versioning
        self.assertEqual(result.model_version, "model_v1")
        self.assertEqual(self.server.current_model_version, "model_v1")
        
        # Run ID constant
        self.assertEqual(self.server.run_id, "run_test")
        for upd in result.updates:
            self.assertEqual(upd.run_id, "run_test")
            self.assertEqual(upd.base_model_version, "model_v0")
            
        # Updates
        self.assertEqual(len(result.updates), 2)
        
        # Model parameters changed
        changed = False
        for k, v in self.global_model.state_dict().items():
            if not torch.allclose(v, original_state[k]):
                changed = True
                break
        self.assertTrue(changed, "Global model parameters should change after aggregation.")
        
        # Evaluator ran
        self.assertIsNotNone(result.evaluation_result)
        self.assertGreater(result.evaluation_result.sample_count, 0)
        
        # History
        self.assertEqual(len(self.server.history), 1)
        self.assertIs(self.server.history[0], result)

    def test_multiple_rounds_sequential(self):
        """Test multiple rounds increment properly and preserve constants."""
        res1 = self.server.execute_round(num_clients=2, seed=1)
        res2 = self.server.execute_round(num_clients=2, seed=2)
        
        self.assertEqual(res1.round_id, 1)
        self.assertEqual(res2.round_id, 2)
        
        self.assertEqual(res1.model_version, "model_v1")
        self.assertEqual(res2.model_version, "model_v2")
        
        # Updates in round 2 should build off model_v1
        for upd in res2.updates:
            self.assertEqual(upd.base_model_version, "model_v1")
            
        self.assertEqual(len(self.server.history), 2)

    def test_invalid_client_count(self):
        """Test rejection of empty or excessive client pool requests."""
        with self.assertRaises(ValueError):
            self.server.execute_round(num_clients=0)
            
        with self.assertRaises(ValueError):
            # We only registered 3 clients
            self.server.execute_round(num_clients=5)

    def test_client_training_failure_halts_round(self):
        """Test that a training failure fails the round clearly without corrupting state."""
        # Create a client that will fail (e.g., malformed batch or incompatible model)
        # We can simulate this by mocking the client's train method
        client_fail = self.client_manager.get_client("client_0")
        
        def failing_train(*args, **kwargs):
            raise RuntimeError("Simulated OOM or training error")
            
        client_fail.train = failing_train # type: ignore
        
        original_state = {k: v.clone() for k, v in self.global_model.state_dict().items()}
        
        with self.assertRaises(RuntimeError) as ctx:
            # Seed guarantees client_0 is selected for reproducibility
            self.server.execute_round(num_clients=3, seed=42)
            
        self.assertIn("Simulated OOM or training error", str(ctx.exception))
        
        # Verify model was not mutated
        for k, v in self.global_model.state_dict().items():
            self.assertTrue(torch.allclose(v, original_state[k]))
            
        # Verify version and history weren't advanced past the failure
        self.assertEqual(self.server.current_round, 0, "Round number should not be consumed on failure.")
        self.assertEqual(self.server.current_model_version, "model_v0")
        self.assertEqual(len(self.server.history), 0)
        
        # Verify no new checkpoint was created
        self.assertFalse(self.server.checkpoint_manager.exists("model_v1"))

    def test_aggregation_failure_halts_round(self):
        """Test that a failure during aggregation halts the round without corrupting state."""
        # Inject a malicious/invalid update to force aggregator to fail
        # by patching the aggregator's method temporarily
        original_aggregate = self.aggregator.aggregate
        
        def failing_aggregate(*args, **kwargs):
            raise ValueError("Incompatible tensor shapes detected!")
            
        self.aggregator.aggregate = failing_aggregate # type: ignore
        
        original_state = {k: v.clone() for k, v in self.global_model.state_dict().items()}
        
        with self.assertRaises(ValueError) as ctx:
            self.server.execute_round(num_clients=2)
            
        self.assertIn("Incompatible tensor shapes detected!", str(ctx.exception))
        
        # Verify model was not mutated
        for k, v in self.global_model.state_dict().items():
            self.assertTrue(torch.allclose(v, original_state[k]))
            
        # Verify version and history weren't advanced past the failure
        self.assertEqual(self.server.current_round, 0, "Round number should not be consumed on aggregation failure.")
        self.assertEqual(self.server.current_model_version, "model_v0")
        self.assertEqual(len(self.server.history), 0)
        
        # Verify no new checkpoint was created
        self.assertFalse(self.server.checkpoint_manager.exists("model_v1"))
        
        # Restore aggregator
        self.aggregator.aggregate = original_aggregate

    def test_missing_eval_dataloader(self):
        """Test execution fails cleanly if no eval dataloader is present."""
        server_no_eval = Server(
            self.global_model,
            self.client_manager,
            self.aggregator,
            self.evaluator,
            "run1"
        )
        with self.assertRaises(ValueError):
            server_no_eval.execute_round(num_clients=1)

    def test_initial_checkpoint_exists(self):
        """Test that the initial model state is checkpointed upon server creation."""
        self.assertTrue(self.server.checkpoint_manager.exists("model_v0"))
        saved = self.server.checkpoint_manager.load("model_v0")
        
        # Should exactly match initial global model
        for k, v in self.global_model.state_dict().items():
            self.assertTrue(torch.allclose(v, saved[k]))

    def test_successful_round_creates_checkpoint(self):
        """Test that a successful round writes a new checkpoint matching the new global state."""
        self.server.execute_round(num_clients=2)
        
        self.assertTrue(self.server.checkpoint_manager.exists("model_v1"))
        saved = self.server.checkpoint_manager.load("model_v1")
        
        # Should exactly match the NEW global model
        for k, v in self.global_model.state_dict().items():
            self.assertTrue(torch.allclose(v, saved[k]))
            
        # The history should show the exact progression
        self.assertEqual(self.server.checkpoint_manager.list_versions(), ["model_v0", "model_v1"])

    def test_evaluation_failure_rolls_back(self):
        """Test that evaluation failure aborts the round, restores weights, and skips checkpointing."""
        # Inject failure into evaluator
        original_eval = self.evaluator.evaluate
        
        def failing_evaluate(*args, **kwargs):
            raise RuntimeError("Evaluation crashed")
            
        self.evaluator.evaluate = failing_evaluate # type: ignore
        
        original_state = {k: v.clone() for k, v in self.global_model.state_dict().items()}
        
        with self.assertRaises(RuntimeError) as ctx:
            self.server.execute_round(num_clients=2)
            
        self.assertIn("Evaluation crashed", str(ctx.exception))
        
        # Verify rollback: global model should be perfectly restored
        for k, v in self.global_model.state_dict().items():
            self.assertTrue(torch.allclose(v, original_state[k]))
            
        # Verify metadata wasn't advanced
        self.assertEqual(self.server.current_round, 0)
        self.assertEqual(self.server.current_model_version, "model_v0")
        self.assertEqual(len(self.server.history), 0)
        
        # Verify NO checkpoint was created
        self.assertFalse(self.server.checkpoint_manager.exists("model_v1"))
        
        # Restore evaluator
        self.evaluator.evaluate = original_eval

if __name__ == '__main__':
    unittest.main()
