import unittest
import torch
from torch.utils.data import TensorDataset, Subset

from core.federated.partitioner import DatasetPartitioner


class TestDatasetPartitioner(unittest.TestCase):
    def setUp(self):
        # Create a synthetic dataset of 100 samples
        self.x_data = torch.randn(100, 1, 28, 28)
        self.y_data = torch.randint(0, 10, (100,))
        self.dataset = TensorDataset(self.x_data, self.y_data)

    def test_partitioner_initialization(self):
        """Test that the partitioner initializes correctly and rejects invalid inputs."""
        partitioner = DatasetPartitioner(num_clients=10, seed=42)
        self.assertEqual(partitioner.num_clients, 10)
        self.assertEqual(partitioner.seed, 42)
        
        with self.assertRaises(ValueError):
            DatasetPartitioner(num_clients=0)
            
        with self.assertRaises(ValueError):
            DatasetPartitioner(num_clients=-5)

    def test_invalid_partition_arguments(self):
        """Test rejection of empty datasets and invalid num_clients ratios."""
        partitioner = DatasetPartitioner(num_clients=10)
        
        # Test empty dataset
        empty_dataset = TensorDataset(torch.empty(0, 1), torch.empty(0))
        with self.assertRaises(ValueError):
            partitioner.partition(empty_dataset)
            
        # Test missing dataset
        with self.assertRaises(ValueError):
            partitioner.partition(None) # type: ignore
            
        # Test too many clients
        too_many_clients = DatasetPartitioner(num_clients=101)
        with self.assertRaises(ValueError):
            too_many_clients.partition(self.dataset)

    def test_correct_number_of_partitions_and_ids(self):
        """Test that it generates the correct number of partitions with stable IDs."""
        partitioner = DatasetPartitioner(num_clients=3)
        partitions = partitioner.partition(self.dataset)
        
        self.assertEqual(len(partitions), 3)
        self.assertIn("client_001", partitions)
        self.assertIn("client_002", partitions)
        self.assertIn("client_003", partitions)
        
        # Verify they are PyTorch Subsets
        for subset in partitions.values():
            self.assertIsInstance(subset, Subset)

    def test_sample_distribution(self):
        """Test that samples are neither lost nor duplicated."""
        partitioner = DatasetPartitioner(num_clients=3)
        partitions = partitioner.partition(self.dataset)
        
        # 100 samples / 3 clients = 34, 33, 33
        self.assertEqual(len(partitions["client_001"]), 34)
        self.assertEqual(len(partitions["client_002"]), 33)
        self.assertEqual(len(partitions["client_003"]), 33)
        
        # Verify no missing or duplicated indices
        all_indices = []
        for subset in partitions.values():
            all_indices.extend(subset.indices) # type: ignore
            
        self.assertEqual(len(all_indices), 100)
        self.assertEqual(len(set(all_indices)), 100)
        
        # Verify it covers 0-99
        self.assertEqual(min(all_indices), 0)
        self.assertEqual(max(all_indices), 99)

    def test_single_client_partition(self):
        """Test behavior when num_clients is exactly 1."""
        partitioner = DatasetPartitioner(num_clients=1)
        partitions = partitioner.partition(self.dataset)
        
        self.assertEqual(len(partitions), 1)
        self.assertIn("client_001", partitions)
        self.assertEqual(len(partitions["client_001"]), 100)

    def test_deterministic_partitioning(self):
        """Test that the same seed produces identical partitions."""
        partitioner1 = DatasetPartitioner(num_clients=5, seed=123)
        partitioner2 = DatasetPartitioner(num_clients=5, seed=123)
        
        part1 = partitioner1.partition(self.dataset)
        part2 = partitioner2.partition(self.dataset)
        
        for client_id in part1.keys():
            self.assertEqual(
                part1[client_id].indices, # type: ignore
                part2[client_id].indices  # type: ignore
            )
            
    def test_nondeterministic_partitioning(self):
        """Test that different seeds produce different partitions."""
        partitioner1 = DatasetPartitioner(num_clients=5, seed=111)
        partitioner2 = DatasetPartitioner(num_clients=5, seed=222)
        
        part1 = partitioner1.partition(self.dataset)
        part2 = partitioner2.partition(self.dataset)
        
        # It's highly improbable that shuffling 100 items with different seeds 
        # results in the exact same arrays for the first client
        self.assertNotEqual(
            part1["client_001"].indices, # type: ignore
            part2["client_001"].indices  # type: ignore
        )

    def test_source_dataset_unmodified(self):
        """Test that the original dataset length and tensors are unchanged."""
        original_length = len(self.dataset)
        original_sum = self.dataset.tensors[1].sum().item()
        
        partitioner = DatasetPartitioner(num_clients=4)
        partitioner.partition(self.dataset)
        
        self.assertEqual(len(self.dataset), original_length)
        self.assertEqual(self.dataset.tensors[1].sum().item(), original_sum)


if __name__ == '__main__':
    unittest.main()
