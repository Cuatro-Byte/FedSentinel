"""
core/sentinel/sentinel.py

FedSentinel — Sentinel Orchestrator (Phase 1 + Phase 2)
Owner: Person 3 — FedSentinel Detection, Impact & Recovery Intelligence

Phase 1: initialise FeatureExtractor; expose extract_features().
Phase 2: initialise StatisticsEngine; expose compute_statistics().

Later phases will add similarity, anomaly detection, reputation,
threat scoring, impact estimation, and recovery.

Contract reference: FEDSENTINEL_NEW_TEAM_ENGINEERING_CONTRACT_v2.md §13
"""

from __future__ import annotations

import logging
from typing import Any

from core.sentinel.feature_extractor import FeatureExtractor, FEATURE_EXTRACTOR_VERSION
from core.sentinel.statistics import StatisticsEngine, STATISTICS_ENGINE_VERSION
from core.sentinel.similarity import SimilarityEngine, SIMILARITY_ENGINE_VERSION
from core.sentinel.anomaly_detector import AnomalyDetector, ANOMALY_ENGINE_VERSION
from core.sentinel.reputation import ReputationEngine, REPUTATION_ENGINE_VERSION
from core.sentinel.threat_scoring import ThreatScoringEngine, THREAT_ENGINE_VERSION
from core.sentinel.impact_estimator import ImpactEstimator, IMPACT_ENGINE_VERSION
from core.sentinel.decision_engine import DecisionEngine, DECISION_ENGINE_VERSION
from core.sentinel.recovery_engine import RecoveryEngine, RECOVERY_ENGINE_VERSION

logger = logging.getLogger(__name__)


class Sentinel:
    """
    FedSentinel Detection, Impact & Recovery Intelligence orchestrator.

    Phase 1 + Phase 2
    ------------------
    Exposes :meth:`extract_features` (Phase 1) and
    :meth:`compute_statistics` (Phase 2). The full detection pipeline
    (similarity, anomaly detection, reputation, threat scoring, impact
    estimation, decision engine, recovery) will be added in subsequent phases.

    Usage
    -----
    >>> sentinel = Sentinel()
    >>> features = sentinel.extract_features(model_update)
    >>> stats    = sentinel.compute_statistics(features)

    Contract
    --------
    - Does NOT accept attack ground truth during inference (§5, Person 3 rules).
    - Does NOT access raw client training data.
    - Does NOT perform database persistence.
    - Does NOT implement frontend logic.
    - Does NOT silently change aggregation behaviour.
    - Does NOT output anomaly scores, threat scores, or response decisions in
      Phase 2 (statistics are descriptive only).
    """

    def __init__(self) -> None:
        """
        Initialise the Sentinel orchestrator.

        Sets up the module-level logger and instantiates
        :class:`~core.sentinel.feature_extractor.FeatureExtractor` and
        :class:`~core.sentinel.statistics.StatisticsEngine`.
        """
        self._logger = logging.getLogger(
            self.__class__.__module__ + "." + self.__class__.__name__
        )
        self._feature_extractor: FeatureExtractor = FeatureExtractor()
        self._statistics_engine: StatisticsEngine = StatisticsEngine()
        self._similarity_engine: SimilarityEngine = SimilarityEngine()
        self._anomaly_detector: AnomalyDetector = AnomalyDetector()
        self._reputation_engine: ReputationEngine = ReputationEngine()
        self._threat_scoring_engine: ThreatScoringEngine = ThreatScoringEngine()
        self._impact_estimator: ImpactEstimator = ImpactEstimator()
        self._decision_engine: DecisionEngine = DecisionEngine()
        self._recovery_engine: RecoveryEngine = RecoveryEngine()
        self._logger.info(
            "Sentinel initialised (feature_extractor_version=%s statistics_version=%s similarity_version=%s anomaly_version=%s reputation_version=%s threat_version=%s impact_version=%s decision_version=%s recovery_version=%s)",
            FEATURE_EXTRACTOR_VERSION,
            STATISTICS_ENGINE_VERSION,
            SIMILARITY_ENGINE_VERSION,
            ANOMALY_ENGINE_VERSION,
            REPUTATION_ENGINE_VERSION,
            THREAT_ENGINE_VERSION,
            IMPACT_ENGINE_VERSION,
            DECISION_ENGINE_VERSION,
            RECOVERY_ENGINE_VERSION,
        )

    # ------------------------------------------------------------------
    # Phase 1 API
    # ------------------------------------------------------------------

    def extract_features(self, update: Any) -> dict[str, Any]:
        """
        Extract numerical features from a canonical ModelUpdate.

        Delegates to :class:`~core.sentinel.feature_extractor.FeatureExtractor`.

        Parameters
        ----------
        update:
            A ``ModelUpdate`` instance conforming to the shared data contract
            (§8.1 of the Engineering Contract).

        Returns
        -------
        dict[str, Any]
            Feature dictionary. See
            :meth:`~core.sentinel.feature_extractor.FeatureExtractor.extract`
            for the complete field list.

        Raises
        ------
        core.sentinel.feature_extractor.FeatureExtractionError
            If the update contains invalid or unextractable parameters.
        """
        self._logger.debug(
            "Sentinel.extract_features called for update_id=%s",
            getattr(update, "update_id", "<unknown>"),
        )
        return self._feature_extractor.extract(update)

    # ------------------------------------------------------------------
    # Phase 2 API
    # ------------------------------------------------------------------

    def compute_statistics(self, features: dict[str, Any]) -> dict[str, Any]:
        """
        Compute descriptive statistics from a FeatureExtractor output dictionary.

        Delegates to :class:`~core.sentinel.statistics.StatisticsEngine`.

        This method performs ANALYSIS ONLY. The returned dictionary contains
        no anomaly scores, threat scores, similarity scores, reputation scores,
        impact scores, or response decisions.

        Parameters
        ----------
        features:
            The dictionary returned by :meth:`extract_features`.

        Returns
        -------
        dict[str, Any]
            Statistics dictionary. See
            :meth:`~core.sentinel.statistics.StatisticsEngine.compute`
            for the complete field list.

        Raises
        ------
        core.sentinel.statistics.StatisticsEngineError
            If the features dictionary is malformed or contains corrupted data.
        """
        self._logger.debug(
            "Sentinel.compute_statistics called for update_id=%s",
            features.get("update_id", "<unknown>"),
        )
        return self._statistics_engine.compute(features)

    # ------------------------------------------------------------------
    # Phase 3 API
    # ------------------------------------------------------------------

    def compute_similarity(self, features_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Compute peer similarity metrics for a list of feature dictionaries.

        Delegates to :class:`~core.sentinel.similarity.SimilarityEngine`.

        Parameters
        ----------
        features_list:
            A list of feature dictionaries from the same round.

        Returns
        -------
        list[dict[str, Any]]
            A list of similarity metric dictionaries, one per client.

        Raises
        ------
        core.sentinel.similarity.SimilarityEngineError
            If updates cannot be safely compared.
        """
        self._logger.debug(
            "Sentinel.compute_similarity called for %d updates",
            len(features_list),
        )
        return self._similarity_engine.compute_similarities(features_list)

    # ------------------------------------------------------------------
    # Phase 4 API
    # ------------------------------------------------------------------

    def compute_anomaly(self, feature: dict[str, Any], statistics: dict[str, Any], similarity: dict[str, Any]) -> dict[str, Any]:
        """
        Compute the composite anomaly score for a single client update.

        Delegates to :class:`~core.sentinel.anomaly_detector.AnomalyDetector`.

        Parameters
        ----------
        feature:
            The feature dictionary produced by Phase 1.
        statistics:
            The statistics dictionary produced by Phase 2.
        similarity:
            The peer similarity dictionary produced by Phase 3.

        Returns
        -------
        dict[str, Any]
            The anomaly dictionary matching the strict Phase 4 Output Contract.

        Raises
        ------
        core.sentinel.anomaly_detector.AnomalyDetectorError
            If inputs are invalid or missing required metrics.
        """
        self._logger.debug(
            "Sentinel.compute_anomaly called for update_id=%s",
            feature.get("update_id", "<unknown>"),
        )
        return self._anomaly_detector.compute_anomaly(feature, statistics, similarity)

    # ------------------------------------------------------------------
    # Phase 5 API
    # ------------------------------------------------------------------

    def update_reputation(self, client_id: str, anomaly_output: dict[str, Any]) -> dict[str, Any]:
        """
        Update the persistent reputation profile for a client.

        Delegates to :class:`~core.sentinel.reputation.ReputationEngine`.

        Parameters
        ----------
        client_id:
            The unique identifier for the client.
        anomaly_output:
            The anomaly dictionary produced by Phase 4.

        Returns
        -------
        dict[str, Any]
            The reputation dictionary matching the strict Phase 5 Output Contract.

        Raises
        ------
        core.sentinel.reputation.ReputationEngineError
            If inputs are invalid or updates are out-of-order.
        """
        self._logger.debug(
            "Sentinel.update_reputation called for client_id=%s",
            client_id,
        )
        return self._reputation_engine.update_reputation(client_id, anomaly_output)

    # ------------------------------------------------------------------
    # Phase 6 API
    # ------------------------------------------------------------------

    def compute_threat(self, 
                       feature: dict[str, Any], 
                       statistics: dict[str, Any], 
                       similarity: dict[str, Any], 
                       anomaly: dict[str, Any], 
                       reputation: dict[str, Any]) -> dict[str, Any]:
        """
        Compute the final composite threat score for a client.
        """
        return self._threat_scoring_engine.compute_threat(
            feature, statistics, similarity, anomaly, reputation
        )

    # ------------------------------------------------------------------
    # Phase 7 API
    # ------------------------------------------------------------------

    def compute_impact(self, 
                       feature: dict[str, Any], 
                       statistics: dict[str, Any], 
                       similarity: dict[str, Any], 
                       anomaly: dict[str, Any], 
                       reputation: dict[str, Any],
                       threat: dict[str, Any]) -> dict[str, Any]:
        """
        Compute the potential impact of a client update.
        """
        return self._impact_estimator.compute_impact(
            feature, statistics, similarity, anomaly, reputation, threat
        )

    # ------------------------------------------------------------------
    # Phase 8 API
    # ------------------------------------------------------------------

    def compute_decision(self,
                         feature: dict[str, Any],
                         threat: dict[str, Any],
                         impact: dict[str, Any],
                         reputation: dict[str, Any],
                         anomaly: dict[str, Any]) -> dict[str, Any]:
        """
        Compute the final recommended security action.
        """
        return self._decision_engine.compute_decision(
            feature, threat, impact, reputation, anomaly
        )

    def simulate_decision(self,
                          feature: dict[str, Any],
                          threat: dict[str, Any],
                          impact: dict[str, Any],
                          reputation: dict[str, Any],
                          anomaly: dict[str, Any]) -> dict[str, Any]:
        """
        Simulate decision computation without side-effects.
        """
        return self._decision_engine.simulate_decision(
            feature, threat, impact, reputation, anomaly
        )

    # ------------------------------------------------------------------
    # Phase 9 API
    # ------------------------------------------------------------------

    def compute_recovery(self,
                         decision: dict[str, Any],
                         threat: dict[str, Any],
                         impact: dict[str, Any],
                         reputation: dict[str, Any]) -> dict[str, Any]:
        """
        Compute the recommended recovery plan.
        """
        return self._recovery_engine.compute_recovery(
            decision, threat, impact, reputation
        )

    def simulate_recovery(self,
                          decision: dict[str, Any],
                          threat: dict[str, Any],
                          impact: dict[str, Any],
                          reputation: dict[str, Any]) -> dict[str, Any]:
        """
        Simulate recovery generation without side-effects.
        """
        return self._recovery_engine.simulate_recovery(
            decision, threat, impact, reputation
        )

    # ------------------------------------------------------------------
    # Phase 10 API - Sentinel Orchestrator
    # ------------------------------------------------------------------

    def run_pipeline(self, update: Any, peer_features: list[dict[str, Any]], round_id: int, simulation_mode: bool = False) -> dict[str, Any]:
        """
        Execute the complete FedSentinel intelligence pipeline for one client.
        """
        import time
        start_time = time.perf_counter()
        failed_stage = None
        pipeline_status = "SUCCESS"
        client_id = getattr(update, "client_id", "unknown")

        res = {
            "feature_output": {},
            "statistics_output": {},
            "similarity_output": {},
            "anomaly_output": {},
            "reputation_output": {},
            "threat_output": {},
            "impact_output": {},
            "decision_output": {},
            "recovery_output": {}
        }

        try:
            # 1. Feature Extraction
            failed_stage = "Feature Extraction"
            f_out = self.extract_features(update)
            res["feature_output"] = f_out

            # 2. Statistics
            failed_stage = "Statistics Engine"
            s_out = self.compute_statistics(f_out)
            res["statistics_output"] = s_out

            # 3. Similarity (Assuming peer_features includes self)
            failed_stage = "Similarity Engine"
            # Note: Similarity usually runs on batch, but here we run it for the single client 
            # by comparing against peers. In run_round, we compute once.
            # To avoid redundant compute, if run_pipeline is called directly we just compute it.
            all_feats = peer_features if peer_features else [f_out]
            if f_out not in all_feats:
                all_feats.append(f_out)
            sim_list = self.compute_similarity(all_feats)
            sim_out = next((s for s in sim_list if s.get("client_id") == client_id), sim_list[0])
            res["similarity_output"] = sim_out

            # 4. Anomaly Detection
            failed_stage = "Anomaly Detection"
            a_out = self.compute_anomaly(f_out, s_out, sim_out)
            res["anomaly_output"] = a_out

            # 5. Reputation Update
            failed_stage = "Reputation Update"
            if simulation_mode:
                # If simulation_mode is true, we must avoid update_reputation if it mutates.
                # Since we don't have simulate_reputation, just pass the existing state translated
                # but it's easier to deepcopy client_states and restore. Wait, run_pipeline doesn't have try/finally.
                # Actually, in run_pipeline we can just call it and rely on simulate_round's wrapper if called from there.
                # Let's just create a dummy ReputationEngine output based on current state.
                state = self._reputation_engine._client_states.get(client_id)
                if state:
                    r_out = {
                        "client_id": client_id,
                        "schema_version": "schema-v1",
                        "reputation_score": state.reputation_score,
                        "reputation_metrics": {"consistency_score": 1.0},
                        "behavior_counts": {"suspicious_count": 0, "malicious_count": 0, "quarantine_count": 0}
                    }
                else:
                    r_out = {
                        "client_id": client_id,
                        "schema_version": "schema-v1",
                        "reputation_score": 1.0,
                        "reputation_metrics": {"consistency_score": 1.0},
                        "behavior_counts": {"suspicious_count": 0, "malicious_count": 0, "quarantine_count": 0}
                    }
            else:
                r_out = self.update_reputation(client_id, a_out)
            res["reputation_output"] = r_out

            # 6. Threat Scoring
            failed_stage = "Threat Scoring"
            t_out = self.compute_threat(f_out, s_out, sim_out, a_out, r_out)
            res["threat_output"] = t_out

            # 7. Impact Estimation
            failed_stage = "Impact Estimation"
            i_out = self.compute_impact(f_out, s_out, sim_out, a_out, r_out, t_out)
            res["impact_output"] = i_out

            # 8. Decision Engine
            failed_stage = "Decision Engine"
            if simulation_mode:
                d_out = self.simulate_decision(f_out, t_out, i_out, r_out, a_out)
            else:
                d_out = self.compute_decision(f_out, t_out, i_out, r_out, a_out)
            res["decision_output"] = d_out

            # 9. Recovery Intelligence
            failed_stage = "Recovery Engine"
            if simulation_mode:
                rec_out = self.simulate_recovery(d_out, t_out, i_out, r_out)
            else:
                rec_out = self.compute_recovery(d_out, t_out, i_out, r_out)
            res["recovery_output"] = rec_out

            failed_stage = None

        except Exception as e:
            self._logger.error("Pipeline failed at stage %s for client %s: %s", failed_stage, client_id, str(e))
            pipeline_status = "FAILED"

        execution_time_ms = (time.perf_counter() - start_time) * 1000.0

        wrapper = {
            "pipeline_version": "sentinel-v1",
            "schema_version": "schema-v1",
            "client_result": res,
            "pipeline_metadata": {
                "round_id": round_id,
                "client_id": client_id,
                "execution_time_ms": execution_time_ms,
                "pipeline_status": pipeline_status,
                "failed_stage": failed_stage
            }
        }
        
        try:
            det_res = self.build_detection_result(client_update, wrapper)
            wrapper["client_result"]["detection_result"] = det_res
        except Exception:
            pass
            
        return wrapper

    def build_detection_result(self, update: Any, pipeline_res: dict[str, Any]) -> dict[str, Any]:
        """
        Builds the canonical DetectionResult object required by Person 1 FL Integration Contract.
        """
        import datetime
        client_res = pipeline_res.get("client_result", {})
        d_out = client_res.get("decision_output", {})
        t_out = client_res.get("threat_output", {})
        a_out = client_res.get("anomaly_output", {})
        r_out = client_res.get("reputation_output", {})
        sim_out = client_res.get("similarity_output", {})
        
        internal_action = d_out.get("recommended_action", "ACCEPT")
        action_map = {
            "ACCEPT": "ACCEPT",
            "MONITOR": "DOWN_WEIGHT",
            "FLAG": "DOWN_WEIGHT",
            "QUARANTINE": "QUARANTINE",
            "ROLLBACK_RECOMMENDED": "QUARANTINE"
        }
        mapped_action = action_map.get(internal_action, "ACCEPT")
        
        threat_score = t_out.get("threat_score", 0.0)
        threat_level = "SAFE"
        if threat_score > 0.8:
            threat_level = "MALICIOUS"
        elif threat_score > 0.4:
            threat_level = "SUSPICIOUS"
            
        feature_summary = {}
        if "feature_output" in client_res:
            f_out = client_res["feature_output"]
            feature_summary = {
                "update_magnitude": f_out.get("update_magnitude", 0.0),
                "parameter_sparsity": f_out.get("parameter_sparsity", 0.0)
            }
            
        return {
            "update_id": getattr(update, "update_id", "unknown"),
            "client_id": pipeline_res.get("pipeline_metadata", {}).get("client_id", getattr(update, "client_id", "unknown")),
            "round_id": pipeline_res.get("pipeline_metadata", {}).get("round_id", getattr(update, "round_id", 0)),
            "threat_score": threat_score,
            "threat_level": threat_level,
            "action": mapped_action,
            "feature_summary": feature_summary,
            "anomaly_score": a_out.get("anomaly_score", 0.0),
<<<<<<< HEAD
            "similarity_score": sim_out.get("similarity_score", 1.0),
=======
            "similarity_score": sim_out.get("consensus", {}).get("consensus_score", 1.0),
>>>>>>> four-way-integration
            "reputation_score": r_out.get("reputation_score", 1.0),
            "explanation_codes": d_out.get("reason_codes", t_out.get("contributing_factors", [])),
            "detector_version": "sentinel-v1",
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

    def run_round(self, client_updates: list[Any], round_id: int) -> list[dict[str, Any]]:
        """
        Process an entire federated round.
        """
        import time
        results = []
        
        # Sort updates to guarantee deterministic ordering
        client_updates = sorted(client_updates, key=lambda x: getattr(x, "client_id", ""))

        # 1-2. Extract and Stat all
        feature_list = []
        stats_dict = {}
        for update in client_updates:
            try:
                f_out = self.extract_features(update)
                feature_list.append(f_out)
                s_out = self.compute_statistics(f_out)
                stats_dict[f_out["client_id"]] = (f_out, s_out)
            except Exception as e:
                print(f"Extraction failed for {getattr(update, 'client_id', 'unknown')}: {e}")

        # 3. Batch Similarity Compute
        sim_list = []
        if feature_list:
            try:
                sim_list = self.compute_similarity(feature_list)
            except Exception:
                pass
        sim_dict = {s.get("client_id"): s for s in sim_list}

        # Complete pipeline per client
        for update in client_updates:
            client_id = getattr(update, "client_id", "unknown")
            start_time = time.perf_counter()
            pipeline_status = "SUCCESS"
            failed_stage = None

            res = {
                "feature_output": {},
                "statistics_output": {},
                "similarity_output": {},
                "anomaly_output": {},
                "reputation_output": {},
                "threat_output": {},
                "impact_output": {},
                "decision_output": {},
                "recovery_output": {}
            }

            try:
                if client_id not in stats_dict:
                    raise ValueError("Failed early extraction")
                
                f_out, s_out = stats_dict[client_id]
                res["feature_output"] = f_out
                res["statistics_output"] = s_out

                failed_stage = "Similarity Engine"
                sim_out = sim_dict.get(client_id)
                if not sim_out:
                    raise ValueError("Similarity missing")
                res["similarity_output"] = sim_out

                failed_stage = "Anomaly Detection"
                a_out = self.compute_anomaly(f_out, s_out, sim_out)
                res["anomaly_output"] = a_out

                failed_stage = "Reputation Update"
                r_out = self.update_reputation(client_id, a_out)
                res["reputation_output"] = r_out

                failed_stage = "Threat Scoring"
                t_out = self.compute_threat(f_out, s_out, sim_out, a_out, r_out)
                res["threat_output"] = t_out

                failed_stage = "Impact Estimation"
                i_out = self.compute_impact(f_out, s_out, sim_out, a_out, r_out, t_out)
                res["impact_output"] = i_out

                failed_stage = "Decision Engine"
                d_out = self.compute_decision(f_out, t_out, i_out, r_out, a_out)
                res["decision_output"] = d_out
                failed_stage = "Recovery Engine"
                rec_out = self.compute_recovery(d_out, t_out, i_out, r_out)
                res["recovery_output"] = rec_out

                failed_stage = None

            except Exception as e:
                print(f"FAILED STAGE: {failed_stage}, Error: {e}")
                pipeline_status = "FAILED"
            
            exec_time = (time.perf_counter() - start_time) * 1000.0
            
            wrapper = {
                "pipeline_version": "sentinel-v1",
                "schema_version": "schema-v1",
                "client_result": res,
                "pipeline_metadata": {
                    "round_id": round_id,
                    "client_id": client_id,
                    "execution_time_ms": exec_time,
                    "pipeline_status": pipeline_status,
                    "failed_stage": failed_stage
                }
            }
            
            # 10. Adapter Layer
            try:
                det_res = self.build_detection_result(update, wrapper)
                wrapper["client_result"]["detection_result"] = det_res
            except Exception as e:
                pass
                
            results.append(wrapper)

        return results

    def simulate_round(self, client_updates: list[Any], round_id: int) -> list[dict[str, Any]]:
        """
        Runs the entire pipeline without updating reputation history.
        """
        import copy
        # Deep copy reputation history to restore after simulation
        backup_states = copy.deepcopy(self._reputation_engine._client_states)
        
        try:
            results = self.run_round(client_updates, round_id)
        finally:
            self._reputation_engine._client_states = backup_states
            
        return results

    def export_pipeline_result(self, result: dict[str, Any]) -> dict[str, Any]:
        """
        Returns fully JSON-serializable dictionary.
        """
        import json
        import numpy as np

        def default_encoder(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        json_str = json.dumps(result, default=default_encoder)
        return json.loads(json_str)
