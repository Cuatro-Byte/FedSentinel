"""
core/sentinel/__init__.py

FedSentinel — Sentinel package public API.
Owner: Person 3 — FedSentinel Detection, Impact & Recovery Intelligence.

Exports all public symbols for Phase 1 (Feature Extraction) and
Phase 2 (Statistics Engine).
"""

from core.sentinel.feature_extractor import (  # noqa: F401
    FEATURE_EXTRACTOR_VERSION,
    SCHEMA_VERSION,
    FeatureExtractionError,
    FeatureExtractor,
)
from core.sentinel.statistics import (  # noqa: F401
    STATISTICS_ENGINE_VERSION,
    StatisticsEngine,
    StatisticsEngineError,
)
from core.sentinel.similarity import (  # noqa: F401
    SIMILARITY_ENGINE_VERSION,
    SimilarityEngine,
    SimilarityEngineError,
)
from core.sentinel.anomaly_detector import (  # noqa: F401
    ANOMALY_ENGINE_VERSION,
    AnomalyDetector,
    AnomalyDetectorError,
)
from core.sentinel.reputation import (  # noqa: F401
    REPUTATION_ENGINE_VERSION,
    ReputationEngine,
    ReputationEngineError,
    ClientReputationState,
)
from core.sentinel.threat_scoring import (  # noqa: F401
    THREAT_ENGINE_VERSION,
    ThreatScoringEngine,
    ThreatScoringError,
)
from core.sentinel.impact_estimator import (  # noqa: F401
    IMPACT_ENGINE_VERSION,
    ImpactEstimator,
    ImpactEstimatorError,
)
from core.sentinel.decision_engine import (  # noqa: F401
    DECISION_ENGINE_VERSION,
    DecisionEngine,
    DecisionEngineError,
)
from core.sentinel.recovery_engine import (  # noqa: F401
    RECOVERY_ENGINE_VERSION,
    RecoveryEngine,
    RecoveryEngineError,
)
from core.sentinel.sentinel import Sentinel  # noqa: F401

__all__ = [
    # Feature Extraction (Phase 1)
    "FEATURE_EXTRACTOR_VERSION",
    "SCHEMA_VERSION",
    "FeatureExtractionError",
    "FeatureExtractor",
    # Statistics Engine (Phase 2)
    "STATISTICS_ENGINE_VERSION",
    "StatisticsEngine",
    "StatisticsEngineError",
    # Similarity Engine (Phase 3)
    "SIMILARITY_ENGINE_VERSION",
    "SimilarityEngine",
    "SimilarityEngineError",
    # Anomaly Detection (Phase 4)
    "ANOMALY_ENGINE_VERSION",
    "AnomalyDetector",
    "AnomalyDetectorError",
    # Client Reputation (Phase 5)
    "REPUTATION_ENGINE_VERSION",
    "ReputationEngine",
    "ReputationEngineError",
    "ClientReputationState",
    # Threat Scoring (Phase 6)
    "THREAT_ENGINE_VERSION",
    "ThreatScoringEngine",
    "ThreatScoringError",
    # Impact Estimation (Phase 7)
    "IMPACT_ENGINE_VERSION",
    "ImpactEstimator",
    "ImpactEstimatorError",
    # Decision Engine (Phase 8)
    "DECISION_ENGINE_VERSION",
    "DecisionEngine",
    "DecisionEngineError",
    # Recovery Intelligence (Phase 9)
    "RECOVERY_ENGINE_VERSION",
    "RecoveryEngine",
    "RecoveryEngineError",
    # Orchestrator
    "Sentinel",
]
