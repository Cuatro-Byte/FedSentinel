import random
from typing import Dict, Optional

from torch.utils.data import Dataset, Subset


class DatasetPartitioner:
    """
    Infrastructure component responsible for splitting a global dataset into
    local datasets suitable for simulated federated-learning clients.
    
    This implementation uses a simple IID random partitioning strategy.
    It does not contain any attack logic or malicious dataset manipulation.
    """

    def __init__(self, num_clients: int, seed: Optional[int] = None):
        """
        Initialize the partitioner.
        
        Args:
            num_clients: The number of clients to divide the dataset among.
            seed: Optional seed for reproducible IID partitioning.
            
        Raises:
            ValueError: If num_clients is less than 1.
        """
        if num_clients < 1:
            raise ValueError("num_clients must be at least 1.")
            
        self.num_clients = num_clients
        self.seed = seed

    def partition(self, dataset: Dataset) -> Dict[str, Dataset]:
        """
        Divide the dataset into local client partitions.
        
        Args:
            dataset: The global PyTorch Dataset to partition.
            
        Returns:
            A dictionary mapping generated client IDs (e.g., 'client_001') 
            to their respective PyTorch Subset datasets.
            
        Raises:
            ValueError: If dataset is missing/empty, or if num_clients exceeds dataset size.
        """
        if dataset is None:
            raise ValueError("A valid PyTorch Dataset must be provided.")
            
        # Some custom datasets might not implement __len__, but standard ones do.
        # This covers the prototype requirement safely.
        try:
            dataset_size = len(dataset) # type: ignore
        except TypeError:
            raise ValueError("Dataset must implement __len__ to be partitionable.")
            
        if dataset_size == 0:
            raise ValueError("Cannot partition an empty dataset.")
            
        if self.num_clients > dataset_size:
            raise ValueError(
                f"Cannot partition {dataset_size} samples among {self.num_clients} clients. "
                "Number of clients cannot exceed number of samples."
            )

        # Create a list of all indices
        indices = list(range(dataset_size))
        
        # Shuffle indices deterministically if a seed is provided
        rng = random.Random(self.seed) if self.seed is not None else random.Random()
        rng.shuffle(indices)

        # Distribute indices as evenly as possible
        # base_size is the minimum number of samples per client
        # remainder is the number of clients that get one extra sample
        base_size = dataset_size // self.num_clients
        remainder = dataset_size % self.num_clients

        partitions: Dict[str, Dataset] = {}
        current_idx = 0

        for i in range(self.num_clients):
            # The first 'remainder' clients get base_size + 1 samples
            # The rest get base_size samples
            chunk_size = base_size + 1 if i < remainder else base_size
            
            client_indices = indices[current_idx : current_idx + chunk_size]
            current_idx += chunk_size
            
            # Use 1-based, zero-padded client IDs (e.g., client_001)
            client_id = f"client_{i + 1:03d}"
            
            # Use Subset to avoid duplicating tensors in memory
            partitions[client_id] = Subset(dataset, client_indices)

        return partitions
