import random
from typing import List, Optional

from core.federated.client import SimulatedClient


class ClientManager:
    """
    Infrastructure component responsible for managing a collection of simulated
    federated learning clients. It handles registration, removal, lookup, and 
    deterministic sampling of clients for FL rounds.
    """

    def __init__(self):
        """Initialize the client manager with an empty collection."""
        # Internal mutable dictionary. Keys are client_ids, values are SimulatedClients.
        self._clients: dict[str, SimulatedClient] = {}

    def register_client(self, client: SimulatedClient) -> None:
        """
        Register a new client.
        
        Args:
            client: The SimulatedClient instance to register.
            
        Raises:
            ValueError: If the client is invalid or if the client_id is already registered.
        """
        if not isinstance(client, SimulatedClient):
            raise ValueError("Can only register SimulatedClient instances.")
        
        client_id = client.client_id
        if client_id in self._clients:
            raise ValueError(f"Client with ID '{client_id}' is already registered.")
            
        self._clients[client_id] = client

    def remove_client(self, client_id: str) -> None:
        """
        Remove an existing client by its ID.
        
        Args:
            client_id: The ID of the client to remove.
            
        Raises:
            ValueError: If the client_id is not found.
        """
        if client_id not in self._clients:
            raise ValueError(f"Cannot remove: client '{client_id}' not found.")
        del self._clients[client_id]

    def get_client(self, client_id: str) -> SimulatedClient:
        """
        Retrieve a registered client by its ID.
        
        Args:
            client_id: The ID of the client to retrieve.
            
        Returns:
            The SimulatedClient instance.
            
        Raises:
            ValueError: If the client_id is not found.
        """
        if client_id not in self._clients:
            raise ValueError(f"Client '{client_id}' not found.")
        return self._clients[client_id]

    def list_clients(self) -> List[SimulatedClient]:
        """
        Retrieve a safe representation of all currently registered clients.
        
        Returns:
            A new list containing the registered SimulatedClient instances,
            deterministically ordered by client_id. This list is safe to mutate 
            without affecting the internal collection.
        """
        # Sort by client_id to guarantee deterministic ordering across runs
        # regardless of the order they were inserted.
        sorted_ids = sorted(self._clients.keys())
        return [self._clients[cid] for cid in sorted_ids]

    def sample_clients(self, num_clients: int, seed: Optional[int] = None) -> List[SimulatedClient]:
        """
        Randomly select a subset of registered clients without replacement.
        
        Args:
            num_clients: The number of clients to sample.
            seed: Optional random seed for reproducible sampling.
            
        Returns:
            A list of sampled SimulatedClient instances.
            
        Raises:
            ValueError: If num_clients is <= 0, or exceeds the number of available clients.
        """
        if num_clients <= 0:
            raise ValueError("Must sample a positive number of clients.")
            
        if num_clients > len(self._clients):
            raise ValueError(
                f"Cannot sample {num_clients} clients; only {len(self._clients)} available."
            )
            
        # Get a deterministic list of IDs
        available_ids = sorted(self._clients.keys())
        
        # Use a local Random instance to preserve determinism without mutating global state
        rng = random.Random(seed) if seed is not None else random.Random()
        
        # Sample without replacement
        sampled_ids = rng.sample(available_ids, num_clients)
        
        return [self._clients[cid] for cid in sampled_ids]
