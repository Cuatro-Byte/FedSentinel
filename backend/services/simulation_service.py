"""
Simulation service — manages simulation lifecycle.

Thin service layer between API routes and the simulation runner.
Business logic belongs here and in the orchestrator, NOT in routes.
"""

import threading
import logging

from sqlalchemy.orm import Session

from backend.adapters.interfaces import FLCoreInterface, AttackInterface, SentinelInterface
from backend.config import AppConfig
from backend.database.repositories import SimulationRepository, AuditRepository
from simulation.runner import SimulationRunner

logger = logging.getLogger("fedsentinel.service.simulation")


class SimulationService:
    """Manages simulation lifecycle — start, status, listing."""

    def __init__(self, db: Session, fl_core: FLCoreInterface,
                 attack_engine: AttackInterface, sentinel: SentinelInterface,
                 config: AppConfig):
        self.db = db
        self.fl_core = fl_core
        self.attack_engine = attack_engine
        self.sentinel = sentinel
        self.config = config
        self.sim_repo = SimulationRepository(db)
        self._active_runs: dict[str, threading.Thread] = {}

    def start_simulation(self, run_id: str, scenario: str | None = None,
                         client_count: int | None = None,
                         rounds: int | None = None,
                         attack_enabled: bool | None = None,
                         attacker_count: int | None = None,
                         intensity: float | None = None,
                         start_round: int | None = None,
                         targets: any = None,
                         seed: int | None = None,
                         background: bool = False) -> dict:
        """Start a new simulation run.

        If background=True, starts asynchronously in a daemon worker thread.
        If background=False, runs synchronously.
        """
        # Apply overrides to config
        if client_count is not None:
            self.config.simulation.client_count = client_count
        if rounds is not None:
            self.config.simulation.rounds = rounds
        if attack_enabled is not None:
            self.config.attack.enabled = attack_enabled
        if scenario is not None:
            self.config.attack.scenario = scenario
        if attacker_count is not None:
            self.config.attack.attacker_count = attacker_count
        if intensity is not None:
            self.config.attack.intensity = intensity
        if start_round is not None:
            self.config.attack.start_round = start_round
        if targets is not None:
            self.config.attack.targets = targets
        if seed is not None:
            self.config.simulation.seed = seed

        scenario_name = scenario or self.config.attack.scenario

        if background:
            # Pre-create simulation record with PENDING status
            self.sim_repo.create(
                run_id=run_id,
                scenario=scenario_name,
                client_count=self.config.simulation.client_count,
                round_count=self.config.simulation.rounds,
                attack_enabled=self.config.attack.enabled,
                seed=self.config.simulation.seed,
            )

            def _background_worker(rid: str, scn: str, cfg: AppConfig):
                from backend.database.database import SessionLocal
                worker_db = SessionLocal()
                try:
                    runner = SimulationRunner(
                        db=worker_db,
                        fl_core=self.fl_core,
                        attack_engine=self.attack_engine,
                        sentinel=self.sentinel,
                        config=cfg,
                    )
                    runner.run_simulation(run_id=rid, scenario=scn)
                except Exception as exc:
                    logger.error("Background simulation %s failed: %s", rid, exc, exc_info=True)
                finally:
                    worker_db.close()

            worker_thread = threading.Thread(
                target=_background_worker,
                args=(run_id, scenario_name, self.config),
                daemon=True,
                name=f"sim-worker-{run_id}",
            )
            self._active_runs[run_id] = worker_thread
            worker_thread.start()
            return {
                "run_id": run_id,
                "simulation_id": run_id,
                "status": "PENDING",
            }

        runner = SimulationRunner(
            db=self.db,
            fl_core=self.fl_core,
            attack_engine=self.attack_engine,
            sentinel=self.sentinel,
            config=self.config,
        )

        runner.run_simulation(run_id=run_id, scenario=scenario)
        return {
            "run_id": run_id,
            "simulation_id": run_id,
            "status": "COMPLETED",
        }

    def get_simulation(self, run_id: str) -> dict | None:
        """Get simulation status and summary."""
        run = self.sim_repo.get(run_id)
        if not run:
            return None
        return {
            "run_id": run.run_id,
            "scenario": run.scenario,
            "client_count": run.client_count,
            "round_count": run.round_count,
            "current_round": run.current_round,
            "status": run.status,
            "model_version": run.model_version,
            "attack_enabled": run.attack_enabled,
            "seed": run.seed,
            "schema_version": run.schema_version,
            "detector_version": run.detector_version,
            "impact_version": run.impact_version,
            "recovery_version": run.recovery_version,
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        }

    def list_simulations(self) -> list[dict]:
        """List all simulation runs."""
        runs = self.sim_repo.list_all()
        return [
            {
                "run_id": r.run_id,
                "scenario": r.scenario,
                "status": r.status,
                "current_round": r.current_round,
                "round_count": r.round_count,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in runs
        ]
