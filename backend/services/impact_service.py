"""
Impact service — impact result lookups.

Provides formatted impact data for API consumption.
Does NOT calculate impact scores — those come from P3.
"""

from sqlalchemy.orm import Session

from backend.database.repositories import ImpactRepository


class ImpactService:
    """Impact result lookups for API routes."""

    def __init__(self, db: Session):
        self.impact_repo = ImpactRepository(db)

    def get_impacts(self, run_id: str) -> list[dict]:
        """Get all impact results for a simulation run."""
        impacts = self.impact_repo.get_by_run(run_id)
        return [
            {
                "update_id": imp.update_id,
                "client_id": imp.client_id,
                "round_id": imp.round_id,
                "impact_score": imp.impact_score,
                "influence_estimate": imp.influence_estimate,
                "parameter_displacement": imp.parameter_displacement,
                "aggregation_weight": imp.aggregation_weight,
                "estimated_accuracy_change": imp.estimated_accuracy_change,
                "estimated_loss_change": imp.estimated_loss_change,
                "impact_level": imp.impact_level,
                "explanation_codes": imp.get_explanation_codes(),
                "impact_breakdown": imp.get_impact_breakdown(),
                "top_impacted_layers": imp.get_top_impacted_layers(),
                "impact_version": imp.impact_version,
                "created_at": imp.created_at.isoformat() if imp.created_at else None,
            }
            for imp in impacts
        ]

    def get_impacts_by_round(self, run_id: str, round_id: int) -> list[dict]:
        """Get impact results for a specific round."""
        impacts = self.impact_repo.get_by_round(run_id, round_id)
        return [
            {
                "update_id": imp.update_id,
                "client_id": imp.client_id,
                "round_id": imp.round_id,
                "impact_score": imp.impact_score,
                "influence_estimate": imp.influence_estimate,
                "parameter_displacement": imp.parameter_displacement,
                "aggregation_weight": imp.aggregation_weight,
                "impact_level": imp.impact_level,
                "explanation_codes": imp.get_explanation_codes(),
                "impact_breakdown": imp.get_impact_breakdown(),
                "top_impacted_layers": imp.get_top_impacted_layers(),
                "impact_version": imp.impact_version,
            }
            for imp in impacts
        ]
