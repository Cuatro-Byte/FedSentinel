"""
Repository layer providing clean CRUD operations for all database entities.

Repositories accept canonical Pydantic models from core/models/ and
persist them as SQLAlchemy ORM records. They do NOT contain business logic.
"""

import json
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from backend.database.models import (
    SimulationRun, ClientRecord, FLRound, ModelUpdateRecord,
    DetectionRecord, ImpactRecord, RecoveryRecord, MetricRecord, AuditEvent,
    ValidationRecord,
)
from core.models import (
    ModelUpdate, DetectionResult, ImpactResult, RecoveryResult, SimulationMetrics,
)


class SimulationRepository:
    """CRUD operations for SimulationRun."""

    def __init__(self, db: Session):
        self.db = db

    def create(self, run_id: str, scenario: str, client_count: int,
               round_count: int, attack_enabled: bool = True,
               seed: int = 42) -> SimulationRun:
        run = SimulationRun(
            run_id=run_id,
            scenario=scenario,
            client_count=client_count,
            round_count=round_count,
            attack_enabled=attack_enabled,
            seed=seed,
            status="PENDING",
            created_at=datetime.utcnow(),
        )
        self.db.add(run)
        self.db.commit()
        self.db.refresh(run)
        return run

    def get(self, run_id: str) -> Optional[SimulationRun]:
        return self.db.query(SimulationRun).filter(
            SimulationRun.run_id == run_id
        ).first()

    def update_status(self, run_id: str, status: str,
                      current_round: int | None = None,
                      model_version: str | None = None) -> Optional[SimulationRun]:
        run = self.get(run_id)
        if run:
            run.status = status
            if current_round is not None:
                run.current_round = current_round
            if model_version is not None:
                run.model_version = model_version
            if status in ("COMPLETED", "FAILED"):
                run.completed_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(run)
        return run

    def list_all(self) -> list[SimulationRun]:
        return self.db.query(SimulationRun).order_by(
            SimulationRun.created_at.desc()
        ).all()


class ClientRepository:
    """CRUD operations for ClientRecord."""

    def __init__(self, db: Session):
        self.db = db

    def create(self, run_id: str, client_id: str) -> ClientRecord:
        client = ClientRecord(
            run_id=run_id,
            client_id=client_id,
            reputation_score=1.0,
            updated_at=datetime.utcnow(),
        )
        self.db.add(client)
        self.db.commit()
        self.db.refresh(client)
        return client

    def create_batch(self, run_id: str, client_ids: list[str]) -> list[ClientRecord]:
        clients = []
        for cid in client_ids:
            c = ClientRecord(
                run_id=run_id,
                client_id=cid,
                reputation_score=1.0,
                updated_at=datetime.utcnow(),
            )
            self.db.add(c)
            clients.append(c)
        self.db.commit()
        for c in clients:
            self.db.refresh(c)
        return clients

    def get_by_run(self, run_id: str) -> list[ClientRecord]:
        return self.db.query(ClientRecord).filter(
            ClientRecord.run_id == run_id
        ).all()

    def get(self, run_id: str, client_id: str) -> Optional[ClientRecord]:
        return self.db.query(ClientRecord).filter(
            ClientRecord.run_id == run_id,
            ClientRecord.client_id == client_id,
        ).first()

    def update_from_detection(self, run_id: str, detection: DetectionResult,
                              impact: ImpactResult | None = None) -> Optional[ClientRecord]:
        client = self.get(run_id, detection.client_id)
        if client:
            client.last_threat_score = detection.threat_score
            client.last_action = detection.action.value
            client.rounds_participated += 1
            if detection.threat_level.value == "SUSPICIOUS":
                client.suspicious_count += 1
            elif detection.threat_level.value == "MALICIOUS":
                client.malicious_count += 1
            if detection.action.value == "QUARANTINE":
                client.quarantine_count += 1
            if impact:
                client.last_impact_score = impact.impact_score
            # Reputation is evidence — updated from detection reputation_score
            client.reputation_score = detection.reputation_score
            client.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(client)
        return client


class RoundRepository:
    """CRUD operations for FLRound."""

    def __init__(self, db: Session):
        self.db = db

    def create(self, run_id: str, round_id: int) -> FLRound:
        fl_round = FLRound(
            run_id=run_id,
            round_id=round_id,
            status="PENDING",
            created_at=datetime.utcnow(),
        )
        self.db.add(fl_round)
        self.db.commit()
        self.db.refresh(fl_round)
        return fl_round

    def get(self, run_id: str, round_id: int) -> Optional[FLRound]:
        return self.db.query(FLRound).filter(
            FLRound.run_id == run_id,
            FLRound.round_id == round_id,
        ).first()

    def get_by_run(self, run_id: str) -> list[FLRound]:
        return self.db.query(FLRound).filter(
            FLRound.run_id == run_id
        ).order_by(FLRound.round_id).all()

    def update(self, fl_round: FLRound, **kwargs) -> FLRound:
        for key, value in kwargs.items():
            if hasattr(fl_round, key):
                setattr(fl_round, key, value)
        self.db.commit()
        self.db.refresh(fl_round)
        return fl_round


class ModelUpdateRepository:
    """CRUD operations for ModelUpdateRecord."""

    def __init__(self, db: Session):
        self.db = db

    def create_from_canonical(self, update: ModelUpdate,
                              fl_round_id: int) -> ModelUpdateRecord:
        record = ModelUpdateRecord(
            update_id=update.update_id,
            run_id=update.run_id,
            round_id=update.round_id,
            fl_round_id=fl_round_id,
            client_id=update.client_id,
            model_version=update.model_version,
            base_model_version=update.base_model_version,
            sample_count=update.sample_count,
            local_loss=update.local_loss,
            local_accuracy=update.local_accuracy,
            training_epochs=update.training_epochs,
            learning_rate=update.learning_rate,
            metadata_json=json.dumps(update.metadata),
            created_at=update.created_at,
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def create_batch(self, updates: list[ModelUpdate],
                     fl_round_id: int) -> list[ModelUpdateRecord]:
        records = []
        for u in updates:
            record = ModelUpdateRecord(
                update_id=u.update_id,
                run_id=u.run_id,
                round_id=u.round_id,
                fl_round_id=fl_round_id,
                client_id=u.client_id,
                model_version=u.model_version,
                base_model_version=u.base_model_version,
                sample_count=u.sample_count,
                local_loss=u.local_loss,
                local_accuracy=u.local_accuracy,
                training_epochs=u.training_epochs,
                learning_rate=u.learning_rate,
                metadata_json=json.dumps(u.metadata),
                created_at=u.created_at,
            )
            self.db.add(record)
            records.append(record)
        self.db.commit()
        for r in records:
            self.db.refresh(r)
        return records

    def get_by_round(self, run_id: str, round_id: int) -> list[ModelUpdateRecord]:
        return self.db.query(ModelUpdateRecord).filter(
            ModelUpdateRecord.run_id == run_id,
            ModelUpdateRecord.round_id == round_id,
        ).all()


class DetectionRepository:
    """CRUD operations for DetectionRecord."""

    def __init__(self, db: Session):
        self.db = db

    def create_from_canonical(self, detection: DetectionResult,
                              run_id: str,
                              fl_round_id: int) -> DetectionRecord:
        record = DetectionRecord(
            update_id=detection.update_id,
            run_id=run_id,
            round_id=detection.round_id,
            fl_round_id=fl_round_id,
            client_id=detection.client_id,
            threat_score=detection.threat_score,
            threat_level=detection.threat_level.value,
            action=detection.action.value,
            anomaly_score=detection.anomaly_score,
            similarity_score=detection.similarity_score,
            reputation_score=detection.reputation_score,
            feature_summary_json=json.dumps(detection.feature_summary),
            explanation_codes_json=json.dumps(detection.explanation_codes),
            detector_version=detection.detector_version,
            created_at=detection.created_at,
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def create_batch(self, detections: list[DetectionResult],
                     run_id: str,
                     fl_round_id: int) -> list[DetectionRecord]:
        records = []
        for d in detections:
            record = DetectionRecord(
                update_id=d.update_id,
                run_id=run_id,
                round_id=d.round_id,
                fl_round_id=fl_round_id,
                client_id=d.client_id,
                threat_score=d.threat_score,
                threat_level=d.threat_level.value,
                action=d.action.value,
                anomaly_score=d.anomaly_score,
                similarity_score=d.similarity_score,
                reputation_score=d.reputation_score,
                feature_summary_json=json.dumps(d.feature_summary),
                explanation_codes_json=json.dumps(d.explanation_codes),
                detector_version=d.detector_version,
                created_at=d.created_at,
            )
            self.db.add(record)
            records.append(record)
        self.db.commit()
        for r in records:
            self.db.refresh(r)
        return records

    def get_by_run(self, run_id: str) -> list[DetectionRecord]:
        return self.db.query(DetectionRecord).filter(
            DetectionRecord.run_id == run_id
        ).order_by(DetectionRecord.round_id).all()

    def get_by_round(self, run_id: str, round_id: int) -> list[DetectionRecord]:
        return self.db.query(DetectionRecord).filter(
            DetectionRecord.run_id == run_id,
            DetectionRecord.round_id == round_id,
        ).all()


class ImpactRepository:
    """CRUD operations for ImpactRecord."""

    def __init__(self, db: Session):
        self.db = db

    def create_from_canonical(self, impact: ImpactResult,
                              run_id: str,
                              fl_round_id: int) -> ImpactRecord:
        record = ImpactRecord(
            update_id=impact.update_id,
            run_id=run_id,
            round_id=impact.round_id,
            fl_round_id=fl_round_id,
            client_id=impact.client_id,
            impact_score=impact.impact_score,
            influence_estimate=impact.influence_estimate,
            parameter_displacement=impact.parameter_displacement,
            aggregation_weight=impact.aggregation_weight,
            estimated_accuracy_change=impact.estimated_accuracy_change,
            estimated_loss_change=impact.estimated_loss_change,
            impact_level=impact.impact_level.value,
            explanation_codes_json=json.dumps(impact.explanation_codes),
            impact_version=impact.impact_version,
            created_at=impact.created_at,
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def create_batch(self, impacts: list[ImpactResult],
                     run_id: str,
                     fl_round_id: int) -> list[ImpactRecord]:
        records = []
        for imp in impacts:
            record = ImpactRecord(
                update_id=imp.update_id,
                run_id=run_id,
                round_id=imp.round_id,
                fl_round_id=fl_round_id,
                client_id=imp.client_id,
                impact_score=imp.impact_score,
                influence_estimate=imp.influence_estimate,
                parameter_displacement=imp.parameter_displacement,
                aggregation_weight=imp.aggregation_weight,
                estimated_accuracy_change=imp.estimated_accuracy_change,
                estimated_loss_change=imp.estimated_loss_change,
                impact_level=imp.impact_level.value if hasattr(imp.impact_level, "value") else str(imp.impact_level),
                explanation_codes_json=json.dumps(imp.explanation_codes or []),
                impact_breakdown_json=json.dumps(getattr(imp, "impact_breakdown", {})),
                top_impacted_layers_json=json.dumps(getattr(imp, "top_impacted_layers", [])),
                impact_version=imp.impact_version,
                created_at=imp.created_at,
            )
            self.db.add(record)
            records.append(record)
        self.db.commit()
        for r in records:
            self.db.refresh(r)
        return records

    def get_by_run(self, run_id: str) -> list[ImpactRecord]:
        return self.db.query(ImpactRecord).filter(
            ImpactRecord.run_id == run_id
        ).order_by(ImpactRecord.round_id).all()

    def get_by_round(self, run_id: str, round_id: int) -> list[ImpactRecord]:
        return self.db.query(ImpactRecord).filter(
            ImpactRecord.run_id == run_id,
            ImpactRecord.round_id == round_id,
        ).all()


class RecoveryRepository:
    """CRUD operations for RecoveryRecord."""

    def __init__(self, db: Session):
        self.db = db

    def create_from_canonical(self, recovery: RecoveryResult) -> RecoveryRecord:
        status_val = recovery.recovery_status.value if hasattr(recovery.recovery_status, "value") else str(recovery.recovery_status)
        record = RecoveryRecord(
            recovery_id=recovery.recovery_id,
            run_id=recovery.run_id,
            round_id=recovery.round_id,
            trigger=recovery.trigger,
            affected_update_ids_json=json.dumps(recovery.affected_update_ids or []),
            excluded_client_ids_json=json.dumps(recovery.excluded_client_ids or []),
            previous_model_version=recovery.previous_model_version,
            recovered_model_version=recovery.recovered_model_version,
            before_accuracy=recovery.before_accuracy,
            after_accuracy=recovery.after_accuracy,
            before_loss=recovery.before_loss,
            after_loss=recovery.after_loss,
            recovery_status=status_val,
            recovery_version=recovery.recovery_version,
            selected_action=getattr(recovery, "selected_action", None),
            details_json=json.dumps(getattr(recovery, "details", {})),
            created_at=recovery.created_at,
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def get_by_run(self, run_id: str) -> list[RecoveryRecord]:
        return self.db.query(RecoveryRecord).filter(
            RecoveryRecord.run_id == run_id
        ).order_by(RecoveryRecord.round_id).all()


class MetricRepository:
    """CRUD operations for MetricRecord."""

    def __init__(self, db: Session):
        self.db = db

    def create_from_canonical(self, metrics: SimulationMetrics) -> MetricRecord:
        record = MetricRecord(
            run_id=metrics.run_id,
            round_id=metrics.round_id,
            model_version=metrics.model_version,
            accuracy=metrics.accuracy,
            loss=metrics.loss,
            attack_success_rate=metrics.attack_success_rate,
            malicious_updates=metrics.malicious_updates,
            suspicious_updates=metrics.suspicious_updates,
            quarantined_updates=metrics.quarantined_updates,
            accepted_updates=metrics.accepted_updates,
            downweighted_updates=metrics.downweighted_updates,
            recovery_triggered=metrics.recovery_triggered,
            recovery_count=metrics.recovery_count,
            created_at=metrics.created_at,
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def get_by_run(self, run_id: str) -> list[MetricRecord]:
        return self.db.query(MetricRecord).filter(
            MetricRecord.run_id == run_id
        ).order_by(MetricRecord.round_id).all()


class AuditRepository:
    """CRUD operations for AuditEvent."""

    def __init__(self, db: Session):
        self.db = db

    def create(self, run_id: str, event_type: str, message: str,
               severity: str = "INFO",
               round_id: int | None = None,
               client_id: str | None = None,
               update_id: str | None = None,
               model_version: str | None = None,
               detector_version: str | None = None,
               impact_version: str | None = None,
               recovery_version: str | None = None) -> AuditEvent:
        event = AuditEvent(
            run_id=run_id,
            round_id=round_id,
            client_id=client_id,
            update_id=update_id,
            event_type=event_type,
            severity=severity,
            message=message,
            model_version=model_version,
            detector_version=detector_version,
            impact_version=impact_version,
            recovery_version=recovery_version,
            timestamp=datetime.utcnow(),
        )
        self.db.add(event)
        self.db.commit()
        self.db.refresh(event)
        return event

    def get_by_run(self, run_id: str) -> list[AuditEvent]:
        return self.db.query(AuditEvent).filter(
            AuditEvent.run_id == run_id
        ).order_by(AuditEvent.timestamp).all()


class ValidationRepository:
    """CRUD operations for ValidationRecord."""

    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        run_id: str,
        round_id: int,
        model_version: Optional[str],
        validation_loss: float,
        validation_accuracy: float,
        loss_spiked: bool,
        baseline_loss: Optional[float],
        baseline_accuracy: Optional[float],
        loss_delta: float,
        accuracy_delta: float,
        validation_status: str,
    ) -> ValidationRecord:
        record = ValidationRecord(
            run_id=run_id,
            round_id=round_id,
            model_version=model_version,
            validation_loss=validation_loss,
            validation_accuracy=validation_accuracy,
            loss_spiked=loss_spiked,
            baseline_loss=baseline_loss,
            baseline_accuracy=baseline_accuracy,
            loss_delta=loss_delta,
            accuracy_delta=accuracy_delta,
            validation_status=validation_status,
            created_at=datetime.utcnow(),
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def get_by_run(self, run_id: str) -> list[ValidationRecord]:
        return self.db.query(ValidationRecord).filter(
            ValidationRecord.run_id == run_id
        ).order_by(ValidationRecord.round_id).all()
