"""
core/attacks/base_attack.py
============================
Abstract base class for all FedSentinel attack simulations.

Contract reference: FedSentinel Team Engineering Contract v2.0 — Person 2
(Adversarial ML / Attack Engineer)

Design principles
-----------------
* Only PyTorch, NumPy, and standard-library modules (``abc``, ``typing``)
  are imported — no other external dependencies.
* Ground-truth attack status (``is_malicious``) is NEVER included in the
  returned objects or passed outside this module boundary.  The ``ModelUpdate``
  schema received by Person 3 (Sentinel) must remain unmodified.
* Concrete attack subclasses override ``apply_data_attack`` and/or
  ``apply_update_attack`` to inject adversarial behaviour.  The base
  implementations are pure identity functions so that the class is safe to
  instantiate for testing without any side-effects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

# PyTorch is listed as a required dependency in requirements.txt.
# Import is deferred to the method signature / type-hint level so that unit
# tests that mock the Dataset type can run without a GPU environment if needed.
try:
    from torch.utils.data import Dataset as TorchDataset  # type: ignore[import]
except ImportError:  # pragma: no cover — guarded import for test environments
    TorchDataset = object  # type: ignore[assignment,misc]


class BaseAttack(ABC):
    """Abstract base class for all FedSentinel adversarial attack modules.

    Every concrete attack must subclass ``BaseAttack`` and is free to override
    :meth:`apply_data_attack`, :meth:`apply_update_attack`, or both, depending
    on the attack category:

    * **Data-plane attacks** (label poisoning, backdoor trigger embedding):
      override :meth:`apply_data_attack`.
    * **Update-plane attacks** (model poisoning, scaling, Byzantine
      perturbations): override :meth:`apply_update_attack`.
    * **Mixed attacks**: override both methods.

    The default (base) implementations are *identity pass-throughs* — they
    return their inputs completely unchanged.  This makes the class safe for
    benign / no-op client simulation and simplifies integration testing.

    Parameters
    ----------
    client_id:
        Unique identifier of the simulated federated client that owns this
        attack instance.  Matches the ``client_id`` field in
        ``core.models.model_update.ModelUpdate``.
    intensity:
        A scalar in ``[0.0, 1.0]`` (or beyond for extreme attacks) that
        controls the strength of the adversarial perturbation.  Concrete
        subclasses are responsible for interpreting this value.
        Default is ``1.0`` (full intensity).
    attack_type:
        A human-readable string tag identifying the attack category, e.g.
        ``"model_poisoning"``, ``"label_poisoning"``, ``"backdoor"``,
        ``"byzantine"``.  Concrete subclasses should override the default
        ``"base"`` value in their own ``__init__`` calls.

    Attributes
    ----------
    client_id : str
    intensity : float
    attack_type : str
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        client_id: str,
        intensity: float = 1.0,
        attack_type: str = "base",
    ) -> None:
        if not isinstance(client_id, str) or not client_id:
            raise ValueError(
                f"client_id must be a non-empty string; got {client_id!r}"
            )
        if not isinstance(intensity, (int, float)):
            raise TypeError(
                f"intensity must be a numeric value; got {type(intensity).__name__}"
            )

        self.client_id: str = client_id
        self.intensity: float = float(intensity)
        self.attack_type: str = attack_type

    # ------------------------------------------------------------------
    # Attack hooks
    # ------------------------------------------------------------------

    def apply_data_attack(self, dataset: "TorchDataset") -> "TorchDataset":
        """Apply a data-plane adversarial transformation to a local dataset.

        This hook is called **before local training** and is intended for
        attacks that corrupt the client's training data, such as:

        * **Label poisoning** — flipping or randomising class labels so that
          the local model learns incorrect decision boundaries.
        * **Backdoor trigger embedding** — inserting a small pattern (e.g. a
          fixed pixel patch) into a subset of training samples and relabelling
          them to a target class, causing the global model to misclassify
          inputs containing the trigger.

        The base implementation is an **identity pass-through**: it returns
        ``dataset`` unchanged.  Concrete attack subclasses that operate at the
        data level must override this method.

        Parameters
        ----------
        dataset:
            The client's local ``torch.utils.data.Dataset`` instance produced
            by ``core.federated.partitioner`` (or equivalent).  The dataset
            object is passed by reference; subclasses may wrap, replace, or
            mutate it as needed.

        Returns
        -------
        torch.utils.data.Dataset
            The (possibly modified) dataset to be used for local training.
            The base implementation returns the original ``dataset`` object
            without modification.

        Notes
        -----
        * Ground-truth attack status (``is_malicious``) must NOT be attached
          to the returned dataset or propagated to any object consumed by
          Person 3's sentinel module.
        * Subclasses should preserve the ``Dataset`` interface so that
          ``torch.utils.data.DataLoader`` can wrap the returned object.
        """
        return dataset

    def apply_update_attack(
        self,
        delta: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Apply an update-plane adversarial transformation to a model delta.

        This hook is called **after local training** and receives the raw
        parameter delta (``w_local − w_global``) expressed as a dictionary
        mapping each parameter name to its NumPy array.  It is intended for
        attacks that manipulate the model update itself, such as:

        * **Model poisoning** — adding a crafted adversarial vector to the
          delta that steers the global model toward a target objective.
        * **Scaling / boosting** — multiplying the delta by a large scalar so
          that the poisoned client exerts disproportionate influence on
          FedAvg aggregation.
        * **Byzantine / sign-flip perturbations** — negating or randomly
          corrupting parameter values to degrade model convergence.

        The base implementation is an **identity pass-through**: it returns
        ``delta`` unchanged.  Concrete attack subclasses that operate at the
        update level must override this method.

        Parameters
        ----------
        delta:
            A dictionary mapping parameter names (e.g. ``"layer1.weight"``)
            to NumPy arrays representing the local model update
            (``w_local − w_global``).  The dictionary is passed by reference;
            subclasses may mutate values in-place or return a new dictionary.

        Returns
        -------
        Dict[str, numpy.ndarray]
            The (possibly modified) parameter delta that will be wrapped into a
            ``ModelUpdate`` and submitted to the federated server.  The base
            implementation returns the original ``delta`` dict unchanged.

        Notes
        -----
        * The returned dictionary must contain the same set of parameter names
          as the input ``delta`` so that the aggregator (Person 1) can apply
          standard FedAvg without additional key-validation logic.
        * Ground-truth attack status (``is_malicious``) must NOT be embedded
          in the returned dictionary or in any object passed to Person 3.
        * Values must be ``numpy.ndarray`` objects; do not return tensors
          directly, as the ``ModelUpdate`` schema stores NumPy arrays.
        """
        return delta

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise attack metadata to a plain Python dictionary.

        Returns
        -------
        dict
            A dictionary with the following keys:

            * ``"client_id"`` (:class:`str`) — the owning client's identifier.
            * ``"attack_type"`` (:class:`str`) — the attack category tag.
            * ``"intensity"`` (:class:`float`) — the attack strength scalar.

        Notes
        -----
        Deliberately minimal: this is safe to log or include in simulation
        audit records.  It does NOT contain any ground-truth labels
        (``is_malicious`` etc.) and must not be extended to include them.
        """
        return {
            "client_id": self.client_id,
            "attack_type": self.attack_type,
            "intensity": self.intensity,
        }

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"client_id={self.client_id!r}, "
            f"attack_type={self.attack_type!r}, "
            f"intensity={self.intensity!r})"
        )
