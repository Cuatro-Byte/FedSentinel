"""
SQLAlchemy ORM models for FedSentinel database entities (Contract §26).

These are DB representations that persist data from the canonical
Pydantic models in core/models/. They are NOT competing definitions.

Entities:
- SimulationRun
- ClientRecord
- FLRound
- ModelUpdateRecord
- DetectionRecord
- ImpactRecord
- RecoveryRecord
- MetricRecord
- AuditEvent
"""

import json
from datetime import datetime

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey
)
from sqlalchemy.orm import relationship

from backend.database.database import Base


class SimulationRun(Base):
    """Simulation run entity (Contract §26 SimulationRun)."""
    __tablename__ = "simulation_runs"

    run_id = Column(String, primary_key=True)
    scenario = Column(String, nullable=False)
    client_count = Column(Integer, nullable=False)
    round_count = Column(Integer, nullable=False)
    attack_enabled = Column(Boolean, default=True)
    seed = Column(Integer, default=42)
    status = Column(String, nullable=False, default="PENDING")
    current_round = Column(Integer, default=0)
    model_version = Column(String, default="model-v0")
    schema_version = Column(String, default="schema-v1")
    detector_version = Column(String, default="detector-v1")
    impact_version = Column(String, default="impact-v1")
    recovery_version = Column(String, default="recovery-v1")
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    rounds = relationship("FLRound", back_populates="simulation_run", cascade="all, delete-orphan")
    clients = relationship("ClientRecord", back_populates="simulation_run", cascade="all, delete-orphan")
    metrics = relationship("MetricRecord", back_populates="simulation_run", cascade="all, delete-orphan")
    recoveries = relationship("RecoveryRecord", back_populates="simulation_run", cascade="all, delete-orphan")
    audit_events = relationship("AuditEvent", back_populates="simulation_run", cascade="all, delete-orphan")


class ClientRecord(Base):
    """Client state within a simulation run."""
    __tablename__ = "clients"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("simulation_runs.run_id"), nullable=False)
    client_id = Column(String, nullable=False)
    reputation_score = Column(Float, default=1.0)
    rounds_participated = Column(Integer, default=0)
    suspicious_count = Column(Integer, default=0)
    malicious_count = Column(Integer, default=0)
    quarantine_count = Column(Integer, default=0)
    last_threat_score = Column(Float, nullable=True)
    last_impact_score = Column(Float, nullable=True)
    last_action = Column(String, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    simulation_run = relationship("SimulationRun", back_populates="clients")


class FLRound(Base):
    """Federated learning round entity (Contract §26 FLRound)."""
    __tablename__ = "fl_rounds"

    id = Column(Integer, primary_key=True, autoincrement=True)
    round_id = Column(Integer, nullable=False)
    run_id = Column(String, ForeignKey("simulation_runs.run_id"), nullable=False)
    model_version = Column(String, nullable=True)
    status = Column(String, default="PENDING")
    update_count = Column(Integer, default=0)
    accepted_count = Column(Integer, default=0)
    suspicious_count = Column(Integer, default=0)
    quarantined_count = Column(Integer, default=0)
    downweighted_count = Column(Integer, default=0)
    recovery_triggered = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    simulation_run = relationship("SimulationRun", back_populates="rounds")
    model_updates = relationship("ModelUpdateRecord", back_populates="fl_round", cascade="all, delete-orphan")
    detections = relationship("DetectionRecord", back_populates="fl_round", cascade="all, delete-orphan")
    impacts = relationship("ImpactRecord", back_populates="fl_round", cascade="all, delete-orphan")


class ModelUpdateRecord(Base):
    """Persisted model update metadata (Contract §26 ModelUpdateRecord).

    Stores metadata about updates. Raw parameter tensors are NOT stored
    in the database for performance reasons — only metadata is persisted.
    """
    __tablename__ = "model_updates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    update_id = Column(String, nullable=False, unique=True)
    run_id = Column(String, nullable=False)
    round_id = Column(Integer, nullable=False)
    fl_round_id = Column(Integer, ForeignKey("fl_rounds.id"), nullable=False)
    client_id = Column(String, nullable=False)
    model_version = Column(String, nullable=True)
    base_model_version = Column(String, nullable=True)
    sample_count = Column(Integer, default=0)
    local_loss = Column(Float, nullable=True)
    local_accuracy = Column(Float, nullable=True)
    training_epochs = Column(Integer, default=1)
    learning_rate = Column(Float, nullable=True)
    metadata_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow)

    fl_round = relationship("FLRound", back_populates="model_updates")

    def get_metadata(self) -> dict:
        """Parse metadata JSON."""
        return json.loads(self.metadata_json) if self.metadata_json else {}


class DetectionRecord(Base):
    """Detection result persistence (Contract §26 DetectionRecord, §9)."""
    __tablename__ = "detection_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    update_id = Column(String, nullable=False)
    run_id = Column(String, nullable=False)
    round_id = Column(Integer, nullable=False)
    fl_round_id = Column(Integer, ForeignKey("fl_rounds.id"), nullable=False)
    client_id = Column(String, nullable=False)
    threat_score = Column(Float, nullable=False)
    threat_level = Column(String, nullable=False)
    action = Column(String, nullable=False)
    anomaly_score = Column(Float, nullable=True)
    similarity_score = Column(Float, nullable=True)
    reputation_score = Column(Float, nullable=True)
    feature_summary_json = Column(Text, default="{}")
    explanation_codes_json = Column(Text, default="[]")
    detector_version = Column(String, default="detector-v1")
    created_at = Column(DateTime, default=datetime.utcnow)

    fl_round = relationship("FLRound", back_populates="detections")

    def get_feature_summary(self) -> dict:
        return json.loads(self.feature_summary_json) if self.feature_summary_json else {}

    def get_explanation_codes(self) -> list:
        return json.loads(self.explanation_codes_json) if self.explanation_codes_json else []


class ImpactRecord(Base):
    """Impact result persistence (Contract §26 ImpactRecord, §10)."""
    __tablename__ = "impact_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    update_id = Column(String, nullable=False)
    run_id = Column(String, nullable=False)
    round_id = Column(Integer, nullable=False)
    fl_round_id = Column(Integer, ForeignKey("fl_rounds.id"), nullable=False)
    client_id = Column(String, nullable=False)
    impact_score = Column(Float, nullable=False)
    influence_estimate = Column(Float, nullable=True)
    parameter_displacement = Column(Float, nullable=True)
    aggregation_weight = Column(Float, nullable=True)
    estimated_accuracy_change = Column(Float, nullable=True)
    estimated_loss_change = Column(Float, nullable=True)
    impact_level = Column(String, nullable=False)
    explanation_codes_json = Column(Text, default="[]")
    impact_version = Column(String, default="impact-v1")
    created_at = Column(DateTime, default=datetime.utcnow)

    fl_round = relationship("FLRound", back_populates="impacts")

    def get_explanation_codes(self) -> list:
        return json.loads(self.explanation_codes_json) if self.explanation_codes_json else []


class RecoveryRecord(Base):
    """Recovery result persistence (Contract §26 RecoveryRecord, §11)."""
    __tablename__ = "recovery_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    recovery_id = Column(String, nullable=False, unique=True)
    run_id = Column(String, ForeignKey("simulation_runs.run_id"), nullable=False)
    round_id = Column(Integer, nullable=False)
    trigger = Column(String, nullable=False)
    affected_update_ids_json = Column(Text, default="[]")
    excluded_client_ids_json = Column(Text, default="[]")
    previous_model_version = Column(String, nullable=True)
    recovered_model_version = Column(String, nullable=True)
    before_accuracy = Column(Float, nullable=True)
    after_accuracy = Column(Float, nullable=True)
    before_loss = Column(Float, nullable=True)
    after_loss = Column(Float, nullable=True)
    recovery_status = Column(String, nullable=False)
    recovery_version = Column(String, default="recovery-v1")
    created_at = Column(DateTime, default=datetime.utcnow)

    simulation_run = relationship("SimulationRun", back_populates="recoveries")

    def get_affected_update_ids(self) -> list:
        return json.loads(self.affected_update_ids_json) if self.affected_update_ids_json else []

    def get_excluded_client_ids(self) -> list:
        return json.loads(self.excluded_client_ids_json) if self.excluded_client_ids_json else []


class MetricRecord(Base):
    """Round-level metrics persistence (Contract §26 MetricRecord, §12)."""
    __tablename__ = "metric_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("simulation_runs.run_id"), nullable=False)
    round_id = Column(Integer, nullable=False)
    model_version = Column(String, nullable=True)
    accuracy = Column(Float, nullable=False)
    loss = Column(Float, nullable=False)
    attack_success_rate = Column(Float, nullable=True)
    malicious_updates = Column(Integer, default=0)
    suspicious_updates = Column(Integer, default=0)
    quarantined_updates = Column(Integer, default=0)
    accepted_updates = Column(Integer, default=0)
    downweighted_updates = Column(Integer, default=0)
    recovery_triggered = Column(Boolean, default=False)
    recovery_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    simulation_run = relationship("SimulationRun", back_populates="metrics")


class AuditEvent(Base):
    """Audit/observability event (Contract §34)."""
    __tablename__ = "audit_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("simulation_runs.run_id"), nullable=False)
    round_id = Column(Integer, nullable=True)
    client_id = Column(String, nullable=True)
    update_id = Column(String, nullable=True)
    event_type = Column(String, nullable=False)
    severity = Column(String, default="INFO")
    message = Column(Text, nullable=False)
    model_version = Column(String, nullable=True)
    detector_version = Column(String, nullable=True)
    impact_version = Column(String, nullable=True)
    recovery_version = Column(String, nullable=True)
    schema_version = Column(String, default="schema-v1")
    timestamp = Column(DateTime, default=datetime.utcnow)

    simulation_run = relationship("SimulationRun", back_populates="audit_events")
