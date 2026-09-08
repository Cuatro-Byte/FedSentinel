"""Runtime adapter registration for the P1, P2, and P3 boundaries."""

from dataclasses import dataclass

from backend.adapters.interfaces import (
    AttackInterface,
    FLCoreInterface,
    SentinelInterface,
)


@dataclass(frozen=True)
class AdapterBundle:
    """Concrete team adapters supplied by the application composition root."""

    fl_core: FLCoreInterface
    attack_engine: AttackInterface
    sentinel: SentinelInterface


_adapter_bundle: AdapterBundle | None = None


def configure_adapters(
    fl_core: FLCoreInterface,
    attack_engine: AttackInterface,
    sentinel: SentinelInterface,
) -> None:
    """Register adapters supplied by the integrated team implementation."""
    global _adapter_bundle
    _adapter_bundle = AdapterBundle(fl_core, attack_engine, sentinel)


def reset_adapters() -> None:
    """Clear registered adapters, primarily for isolated test lifecycles."""
    global _adapter_bundle
    _adapter_bundle = None


def get_adapters() -> AdapterBundle:
    """Return configured adapters or fail with an actionable integration error."""
    if _adapter_bundle is None:
        raise RuntimeError(
            "FedSentinel adapters are not configured. "
            "Register the P1, P2, and P3 implementations before starting the API."
        )
    return _adapter_bundle
