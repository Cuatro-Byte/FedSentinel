"""
Simulation runner — the core orchestration loop.

Owned by Person 4 (Contract §5).

This module COORDINATES calls to P1, P2, P3 interfaces and persists results.
It does NOT implement any FL training, attack, detection, impact estimation,
recovery intelligence, or aggregation algorithms.

Orchestration flow (Contract §57, corrected checklist §7):
    P1 training
    → P2 attack injection
    → P3 detection
    → P3 impact estimation
    → P3 response decision
    → P1 trust-aware aggregation (P4 passes actions to P1)
    → P1 evaluation
    → P3 recovery decision if required
    → P1 selective re-aggregation if triggered
    → P4 persistence/API
"""

import logging
from typing import Any
import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from backend.adapters.interfaces import FLCoreInterface, AttackInterface, SentinelInterface
from backend.config import AppConfig
from backend.database.repositories import (
    SimulationRepository, ClientRepository, RoundRepository,
    ModelUpdateRepository, DetectionRepository, ImpactRepository,
    RecoveryRepository, MetricRepository, AuditRepository,
    ValidationRepository,
)
from core.models import SimulationMetrics
from core.models.enums import ThreatLevel, ResponseAction, RecoveryStatus
from core.validation.client_validator import ClientUpdateFirewall
from simulation.scenario_manager import ScenarioManager

logger = logging.getLogger("fedsentinel.runner")


def _format_float(val: Any, precision: int = 4) -> str:
    """Safely format a float or return 'N/A' if None/invalid, preventing format crashes."""
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.{precision}f}"
    except (ValueError, TypeError):
        return "N/A"


class SimulationRunner:
    """Orchestrates a full FedSentinel simulation run.

    Coordinates P1, P2, P3 via their interfaces.
    Persists all results to the database.
    Records audit events for observability.
    """

    def __init__(
        self,
        db: Session,
        fl_core: FLCoreInterface,
        attack_engine: AttackInterface,
        sentinel: SentinelInterface,
        config: AppConfig,
    ):
        self.db = db
        self.fl_core = fl_core
        self.attack_engine = attack_engine
        self.sentinel = sentinel
        self.config = config
        self.scenario = ScenarioManager(config)
        self.client_firewall = ClientUpdateFirewall()

        # Repositories
        self.sim_repo = SimulationRepository(db)
        self.client_repo = ClientRepository(db)
        self.round_repo = RoundRepository(db)
        self.update_repo = ModelUpdateRepository(db)
        self.detection_repo = DetectionRepository(db)
        self.impact_repo = ImpactRepository(db)
        self.recovery_repo = RecoveryRepository(db)
        self.metric_repo = MetricRepository(db)
        self.audit_repo = AuditRepository(db)
        self.val_repo = ValidationRepository(db)

    def run_simulation(self, run_id: str | None = None,
                       scenario: str | None = None) -> str:
        """Execute a complete multi-round simulation.

        Args:
            run_id: Optional run ID. Generated if not provided.
            scenario: Optional scenario name override.

        Returns:
            The run_id of the completed simulation.
        """
        if run_id is None:
            run_id = f"RUN-{uuid.uuid4().hex[:8].upper()}"

        sim_params = self.scenario.get_simulation_params()
        attack_params = self.scenario.get_attack_params()
        recovery_params = self.scenario.get_recovery_params()
        fl_config = self.scenario.get_fl_config()
        client_ids = self.scenario.get_client_ids()
        total_rounds = sim_params["rounds"]
        scenario_name = scenario or attack_params.get("scenario", "normal")
        logger.info(f"Preparing simulation {run_id} ({scenario_name})")

        # Reset stateful adapters (Issue 1)
        if hasattr(self.sentinel, 'reset'):
            self.sentinel.reset()

        # 1. Create or get simulation run
        sim_run = self.sim_repo.get(run_id)
        if not sim_run:
            sim_run = self.sim_repo.create(
                run_id=run_id,
                scenario=scenario_name,
                client_count=len(client_ids),
                round_count=total_rounds,
                attack_enabled=attack_params.get("enabled", True),
                seed=sim_params.get("seed", 42),
            )
        self.sim_repo.update_status(run_id, "RUNNING")

        # 2. Create client records
        self.client_repo.create_batch(run_id, client_ids)

        # 3. Provision P1 clients and initialize global model
        if hasattr(self.fl_core, "provision_clients"):
            self.fl_core.provision_clients(
                client_count=len(client_ids),
                seed=sim_params.get("seed", 42),
                local_epochs=fl_config.get("local_epochs", 2),
            )
        fl_config["client_count"] = len(client_ids)
        model_version = self.fl_core.initialize_global_model(fl_config)
        self.sim_repo.update_status(run_id, "RUNNING", model_version=model_version)

        self.audit_repo.create(
            run_id=run_id,
            event_type="SIMULATION_STARTED",
            message=f"Simulation started: {total_rounds} rounds, "
                    f"{len(client_ids)} clients, scenario={scenario_name}",
            severity="INFO",
            model_version=model_version,
        )

        logger.info(f"Simulation {run_id} started: {total_rounds} rounds, "
                     f"{len(client_ids)} clients")

        try:
            # 4. Run each round
            for round_num in range(1, total_rounds + 1):
                model_version = self._run_round(
                    run_id=run_id,
                    round_id=round_num,
                    model_version=model_version,
                    client_ids=client_ids,
                    fl_config=fl_config,
                    attack_params=attack_params,
                    recovery_params=recovery_params,
                )
                self.sim_repo.update_status(
                    run_id, "RUNNING",
                    current_round=round_num,
                    model_version=model_version,
                )

            # 5. Complete
            self.sim_repo.update_status(run_id, "COMPLETED", model_version=model_version)
            self.audit_repo.create(
                run_id=run_id,
                event_type="SIMULATION_COMPLETED",
                message=f"Simulation completed successfully. Final model: {model_version}",
                severity="INFO",
                model_version=model_version,
            )
            logger.info(f"Simulation {run_id} completed. Final model: {model_version}")

        except Exception as e:
            # A failed flush leaves SQLAlchemy sessions in a pending-rollback
            # state.  Reset it before recording the failed run and audit event.
            self.db.rollback()
            self.sim_repo.update_status(run_id, "FAILED")
            self.audit_repo.create(
                run_id=run_id,
                event_type="SIMULATION_FAILED",
                message=f"Simulation failed: {str(e)}",
                severity="ALERT",
            )
            logger.error(f"Simulation {run_id} failed: {e}")
            raise

        return run_id

    def _run_round(
        self,
        run_id: str,
        round_id: int,
        model_version: str,
        client_ids: list[str],
        fl_config: dict,
        attack_params: dict,
        recovery_params: dict,
    ) -> str:
        """Execute a single FL round. Returns the (possibly recovered) model version."""

        logger.info(f"  Round {round_id} starting (model={model_version})")

        # Create round record
        fl_round = self.round_repo.create(run_id, round_id)

        self.audit_repo.create(
            run_id=run_id, round_id=round_id,
            event_type="ROUND_STARTED",
            message=f"Round {round_id} started",
            severity="INFO",
            model_version=model_version,
        )

        # Step 1: Pre-training data-plane attacks (P2 -> client dataset -> P1)
        original_dataloaders = {}
        if hasattr(self.fl_core, "client_manager") and hasattr(self.attack_engine, "apply_data_attack"):
            from torch.utils.data import DataLoader
            for cid in client_ids:
                client = self.fl_core.client_manager.get_client(cid)
                if client and hasattr(client, "dataloader"):
                    original_dataset = getattr(client.dataloader, "dataset", None)
                    if original_dataset is not None:
                        modified_dataset = self.attack_engine.apply_data_attack(
                            client_id=cid,
                            dataset=original_dataset,
                            round_id=round_id,
                            config=attack_params,
                            client_ids=client_ids,
                        )
                        if modified_dataset is not original_dataset:
                            original_dataloaders[cid] = client.dataloader
                            batch_size = getattr(client.dataloader, "batch_size", 16) or 16
                            client.dataloader = DataLoader(modified_dataset, batch_size=batch_size, shuffle=True)

        try:
            # Step 2: P1 trains clients
            updates = self.fl_core.train_clients(
                run_id, round_id, model_version, client_ids, fl_config
            )
        finally:
            # Restore original clean client dataloaders
            if hasattr(self.fl_core, "client_manager"):
                for cid, orig_dl in original_dataloaders.items():
                    client = self.fl_core.client_manager.get_client(cid)
                    if client:
                        client.dataloader = orig_dl

        # Step 3: P2 applies update-level attacks
        updates = self.attack_engine.apply_attacks(updates, round_id, attack_params)

        # Persist model updates (metadata only)
        self.update_repo.create_batch(updates, fl_round.id)

        # Step 3.5: Client-Update Validation Firewall
        ref_keys = set(self.fl_core.global_model.state_dict().keys()) if hasattr(self.fl_core, "global_model") and hasattr(self.fl_core.global_model, "state_dict") else None
        ref_shapes = {k: tuple(v.shape) for k, v in self.fl_core.global_model.state_dict().items()} if hasattr(self.fl_core, "global_model") and hasattr(self.fl_core.global_model, "state_dict") else None

        valid_updates, fw_detections, fw_impacts = self.client_firewall.validate_updates(
            updates, round_id=round_id, expected_keys=ref_keys, expected_shapes=ref_shapes
        )

        # Step 4: P3 detection (executed only on structurally and numerically valid updates)
        if valid_updates:
            sentinel_detections = self.sentinel.detect(valid_updates, round_id)
        else:
            sentinel_detections = []

        fw_det_map = {d.update_id: d for d in fw_detections}
        sen_det_map = {d.update_id: d for d in sentinel_detections}
        detections = [fw_det_map.get(u.update_id) or sen_det_map[u.update_id] for u in updates]
        self.detection_repo.create_batch(detections, run_id, fl_round.id)

        # Step 5: P3 impact estimation (executed only on valid updates)
        if valid_updates:
            sentinel_impacts = self.sentinel.estimate_impact(valid_updates, sentinel_detections, round_id)
        else:
            sentinel_impacts = []

        fw_imp_map = {i.update_id: i for i in fw_impacts}
        sen_imp_map = {i.update_id: i for i in sentinel_impacts}
        impacts = [fw_imp_map.get(u.update_id) or sen_imp_map[u.update_id] for u in updates]
        self.impact_repo.create_batch(impacts, run_id, fl_round.id)

        # Step 6: P3 response decision
        actions = self.sentinel.decide_response(detections, impacts)

        # Ground truth is captured only after inference and response decisions.
        attacker_ids = set(
            self.attack_engine.get_attacker_ids(round_id, attack_params)
        ) if attack_params.get("enabled", True) else set()

        # Update client records from detection/impact results
        detection_map = {d.update_id: d for d in detections}
        impact_map = {imp.update_id: imp for imp in impacts}
        for update in updates:
            det = detection_map.get(update.update_id)
            imp = impact_map.get(update.update_id)
            if det:
                self.client_repo.update_from_detection(run_id, det, imp)

        # Count actions for round record
        accepted = sum(1 for a in actions.values() if a == ResponseAction.ACCEPT)
        downweighted = sum(1 for a in actions.values() if a == ResponseAction.DOWN_WEIGHT)
        quarantined = sum(1 for a in actions.values() if a == ResponseAction.QUARANTINE)
        suspicious = sum(1 for d in detections if d.threat_level == ThreatLevel.SUSPICIOUS)
        malicious = sum(1 for d in detections if d.threat_level == ThreatLevel.MALICIOUS)
        attacked_updates = [update for update in updates if update.client_id in attacker_ids]
        accepted_attacks = sum(
            1 for update in attacked_updates
            if actions.get(update.update_id) == ResponseAction.ACCEPT
        )
        attack_success_rate = (
            accepted_attacks / len(attacked_updates)
            if attacked_updates else None
        )

        # Log quarantine events
        for det in detections:
            if det.action == ResponseAction.QUARANTINE:
                self.audit_repo.create(
                    run_id=run_id, round_id=round_id,
                    client_id=det.client_id,
                    update_id=det.update_id,
                    event_type="UPDATE_QUARANTINED",
                    message=f"Client {det.client_id} quarantined: "
                            f"threat={det.threat_score:.2f}, "
                            f"reasons={det.explanation_codes}",
                    severity="WARN",
                    detector_version=det.detector_version,
                )

        # Capture the pre-round base version BEFORE aggregation produces the candidate.
        # This is the W_base that every client delta was computed against, and the
        # correct starting point for selective recovery re-aggregation.
        pre_round_model_version = model_version

        # Step 7: P1 trust-aware aggregation (P4 passes detections to P1)
        new_model_version = self.fl_core.aggregate(
            updates, detections, model_version
        )

        # ServerValidationGate: validate the candidate before treating as final
        val_metrics, loss_spiked = None, False
        if hasattr(self.fl_core, "validate_candidate"):
            val_metrics, loss_spiked = self.fl_core.validate_candidate(new_model_version)
            if val_metrics is not None:
                val_loss = float(val_metrics.get("val_loss", 0.0))
                val_acc = float(val_metrics.get("val_acc", 0.0))
                loss_delta = float(val_metrics.get("loss_delta", 0.0))
                acc_delta = float(val_metrics.get("acc_delta", 0.0))
                val_status = "ANOMALOUS" if loss_spiked else "HEALTHY"
                # Compute baseline if previous baseline was updated
                baseline_loss = val_loss - loss_delta if loss_delta != 0.0 else val_loss
                baseline_acc = val_acc + acc_delta if acc_delta != 0.0 else val_acc
                self.val_repo.create(
                    run_id=run_id,
                    round_id=round_id,
                    model_version=new_model_version,
                    validation_loss=val_loss,
                    validation_accuracy=val_acc,
                    loss_spiked=loss_spiked,
                    baseline_loss=baseline_loss,
                    baseline_accuracy=baseline_acc,
                    loss_delta=loss_delta,
                    accuracy_delta=acc_delta,
                    validation_status=val_status,
                )

        # Step 8: P1 evaluation
        evaluation = self.fl_core.evaluate(new_model_version)

        # Step 9: P3 recovery decision
        recovery_triggered = False
        recovery_count = 0
        recovery_result = self.sentinel.check_recovery(
            evaluation, detections, impacts,
            run_id, round_id, new_model_version,
            recovery_params,
            val_metrics=val_metrics,
            loss_spiked=loss_spiked
        )

        # Step 10: If recovery triggered, P1 re-aggregates
        if recovery_result and recovery_result.recovery_status == RecoveryStatus.TRIGGERED:
            recovery_triggered = True
            recovery_count = 1

            self.audit_repo.create(
                run_id=run_id, round_id=round_id,
                event_type="RECOVERY_TRIGGERED",
                message=f"Recovery triggered: {recovery_result.trigger}. "
                        f"Excluding clients: {recovery_result.excluded_client_ids}",
                severity="ALERT",
                model_version=new_model_version,
                recovery_version=recovery_result.recovery_version,
            )

            # P1 performs selective re-aggregation.
            # IMPORTANT: pass pre_round_model_version (W_base), NOT new_model_version
            # (the candidate W_candidate = W_base + ΔW_all).  Using the candidate would
            # apply deltas twice: W_candidate + ΔW_recovered = W_base + ΔW_all + ΔW_recovered.
            # The correct semantic is: W_recovered = W_base + ΔW_recovered.
            recovered_model_version = self.fl_core.re_aggregate(
                updates,
                recovery_result.excluded_client_ids,
                pre_round_model_version,
            )

            # P1 evaluates recovered model
            recovered_eval = self.fl_core.evaluate(recovered_model_version)

            # Update recovery result with actual recovered data
            recovery_result = recovery_result.model_copy(update={
                "recovered_model_version": recovered_model_version,
                "after_accuracy": recovered_eval.get("accuracy"),
                "after_loss": recovered_eval.get("loss"),
                "recovery_status": RecoveryStatus.COMPLETED,
            })

            new_model_version = recovered_model_version
            evaluation = recovered_eval

            before_acc_str = _format_float(recovery_result.before_accuracy)
            after_acc_str = _format_float(recovered_eval.get("accuracy") if recovered_eval else None)

            self.audit_repo.create(
                run_id=run_id, round_id=round_id,
                event_type="RECOVERY_COMPLETED",
                message=f"Recovery completed. "
                        f"Before: acc={before_acc_str}, "
                        f"After: acc={after_acc_str}. "
                        f"New model: {recovered_model_version}",
                severity="INFO",
                model_version=recovered_model_version,
                recovery_version=recovery_result.recovery_version,
            )

        # Persist recovery result
        if recovery_result is not None:
            self.recovery_repo.create_from_canonical(recovery_result)

        # Update round record
        self.round_repo.update(
            fl_round,
            status="COMPLETED",
            model_version=new_model_version,
            update_count=len(updates),
            accepted_count=accepted,
            suspicious_count=suspicious,
            quarantined_count=quarantined,
            downweighted_count=downweighted,
            recovery_triggered=recovery_triggered,
        )

        # Step 11: Assemble and persist SimulationMetrics (P4 responsibility)
        eval_acc = evaluation.get("accuracy") if evaluation else None
        eval_loss = evaluation.get("loss") if evaluation else None
        metrics = SimulationMetrics(
            run_id=run_id,
            round_id=round_id,
            model_version=new_model_version,
            accuracy=float(eval_acc) if eval_acc is not None else 0.0,
            loss=float(eval_loss) if eval_loss is not None else 0.0,
            attack_success_rate=attack_success_rate,
            malicious_updates=malicious,
            suspicious_updates=suspicious,
            quarantined_updates=quarantined,
            accepted_updates=accepted,
            downweighted_updates=downweighted,
            recovery_triggered=recovery_triggered,
            recovery_count=recovery_count,
        )
        self.metric_repo.create_from_canonical(metrics)

        acc_str = _format_float(evaluation.get("accuracy") if evaluation else None)
        loss_str = _format_float(evaluation.get("loss") if evaluation else None)
        self.audit_repo.create(
            run_id=run_id, round_id=round_id,
            event_type="ROUND_COMPLETED",
            message=f"Round {round_id} completed: "
                    f"acc={acc_str}, "
                    f"loss={loss_str}, "
                    f"model={new_model_version}",
            severity="INFO",
            model_version=new_model_version,
        )

        logger.info(f"  Round {round_id} completed: "
                     f"acc={acc_str}, "
                     f"model={new_model_version}")

        return new_model_version
