import unittest
import torch
from torch.utils.data import TensorDataset, DataLoader
from core.federated.trainer import LocalTrainer
from core.federated.client import SimulatedClient
from core.federated.client_manager import ClientManager

class TestClientManager(unittest.TestCase):
    def setUp(self):
        self.manager = ClientManager()
        
        # Helper to create a dummy client
        self.trainer = LocalTrainer(learning_rate=0.01, epochs=1, device="cpu")
        dataset = TensorDataset(torch.randn(1, 1, 28, 28), torch.randint(0, 10, (1,)))
        self.dataloader = DataLoader(dataset)
        
    def _create_client(self, client_id: str) -> SimulatedClient:
        return SimulatedClient(client_id, self.dataloader, self.trainer)

    def test_manager_initialization(self):
        """Test that the manager initializes correctly."""
        self.assertEqual(len(self.manager.list_clients()), 0)

    def test_register_client(self):
        """Test registering a single and multiple clients."""
        client1 = self._create_client("client_1")
        self.manager.register_client(client1)
        self.assertEqual(len(self.manager.list_clients()), 1)
        
        client2 = self._create_client("client_2")
        self.manager.register_client(client2)
        self.assertEqual(len(self.manager.list_clients()), 2)

    def test_duplicate_registration_fails(self):
        """Test that duplicate registration raises an error."""
        client1 = self._create_client("client_1")
        self.manager.register_client(client1)
        
        # Even if it's a different instance, the same ID should fail
        client1_duplicate = self._create_client("client_1")
        with self.assertRaises(ValueError):
            self.manager.register_client(client1_duplicate)

    def test_lookup_and_missing_client(self):
        """Test getting a client and requesting a missing one."""
        client1 = self._create_client("client_1")
        self.manager.register_client(client1)
        
        retrieved = self.manager.get_client("client_1")
        self.assertIs(retrieved, client1)
        
        with self.assertRaises(ValueError):
            self.manager.get_client("missing_client")

    def test_remove_client(self):
        """Test removing a client and attempting to remove a missing one."""
        client1 = self._create_client("client_1")
        self.manager.register_client(client1)
        
        self.manager.remove_client("client_1")
        self.assertEqual(len(self.manager.list_clients()), 0)
        
        with self.assertRaises(ValueError):
            self.manager.remove_client("client_1")

    def test_list_clients_safety(self):
        """Test list_clients returns all clients safely."""
        client1 = self._create_client("c1")
        client2 = self._create_client("c2")
        self.manager.register_client(client2)
        self.manager.register_client(client1)
        
        clients = self.manager.list_clients()
        self.assertEqual(len(clients), 2)
        # Verify deterministic ordering by ID
        self.assertEqual(clients[0].client_id, "c1")
        self.assertEqual(clients[1].client_id, "c2")
        
        # Verify mutation doesn't affect manager
        clients.pop()
        self.assertEqual(len(self.manager.list_clients()), 2)

    def test_sampling_clients(self):
        """Test valid sampling requests."""
        for i in range(5):
            self.manager.register_client(self._create_client(f"client_{i}"))
            
        sampled = self.manager.sample_clients(3)
        self.assertEqual(len(sampled), 3)
        # Ensure without replacement (all unique)
        sampled_ids = {c.client_id for c in sampled}
        self.assertEqual(len(sampled_ids), 3)
        
        # Verify sampling doesn't modify the pool
        self.assertEqual(len(self.manager.list_clients()), 5)

    def test_invalid_sampling_requests(self):
        """Test invalid sampling limits."""
        client1 = self._create_client("client_1")
        self.manager.register_client(client1)
        
        with self.assertRaises(ValueError):
            self.manager.sample_clients(0)
            
        with self.assertRaises(ValueError):
            self.manager.sample_clients(-1)
            
        with self.assertRaises(ValueError):
            self.manager.sample_clients(2)

    def test_deterministic_sampling(self):
        """Test that sampling with the same seed is reproducible."""
        for i in range(10):
            self.manager.register_client(self._create_client(f"client_{i}"))
            
        seed = 42
        sampled1 = self.manager.sample_clients(4, seed=seed)
        sampled2 = self.manager.sample_clients(4, seed=seed)
        
        ids1 = [c.client_id for c in sampled1]
        ids2 = [c.client_id for c in sampled2]
        
        self.assertEqual(ids1, ids2)
        
        # Different seed produces different results (highly likely with N=4 from 10)
        sampled3 = self.manager.sample_clients(4, seed=99)
        ids3 = [c.client_id for c in sampled3]
        self.assertNotEqual(ids1, ids3)

if __name__ == '__main__':
    unittest.main()
