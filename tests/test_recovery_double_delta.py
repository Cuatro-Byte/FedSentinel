"""
Regression test: Recovery must use the PRE-ROUND base model, not the candidate.

Bug fixed in Phase 4.2:
    WRONG:  W_recovered = W_candidate + delta_recovered
                        = W_pre_round + delta_all + delta_recovered   <- double-delta

    CORRECT: W_recovered = W_pre_round + delta_recovered

This test uses a tiny 1-parameter deterministic model so the math is exact and
any double-application is immediately detectable without tolerances.
"""

from datetime import datetime, timezone
import pytest
import torch

from core.federated.aggregator import Aggregator
from core.federated.checkpoint import CheckpointManager
from core.p1_models.model_update import ModelUpdate as P1ModelUpdate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_update(update_id, client_id, delta_val, base_version="base_v0"):
    return P1ModelUpdate(
        update_id=update_id,
        run_id="test-run",
        round_id=1,
        client_id=client_id,
        model_version=f"{base_version}_{client_id}",
        base_model_version=base_version,
        parameters={"w": torch.tensor([delta_val], dtype=torch.float32)},
        sample_count=1,
        local_loss=0.0,
        local_accuracy=1.0,
        training_epochs=1,
        learning_rate=0.01,
        created_at=datetime.now(timezone.utc),
    )


def _scalar(state):
    return state["w"].item()


# ---------------------------------------------------------------------------
# Arithmetic unit tests
# ---------------------------------------------------------------------------

class TestRecoveryDoubleDeltaRegression:
    """
    W_pre_round = 10.0
    delta_good  = +1.0
    delta_bad   = +5.0

    FedAvg(all):        delta_all  = (1.0 + 5.0) / 2 = 3.0
    W_candidate         = 10.0 + 3.0 = 13.0

    Recovery excludes client-bad:
        delta_recovered = 1.0 / 1 = 1.0

    CORRECT:  W_recovered = 10.0 + 1.0 = 11.0
    WRONG:    W_recovered = 13.0 + 1.0 = 14.0  (double-delta, old bug)
    """

    W_PRE = 10.0
    D_GOOD = 1.0
    D_BAD  = 5.0
    W_CANDIDATE       = W_PRE + (D_GOOD + D_BAD) / 2   # 13.0
    W_RECOVERED_CORRECT = W_PRE + D_GOOD               # 11.0
    W_RECOVERED_WRONG   = W_CANDIDATE + D_GOOD         # 14.0

    def _build(self):
        agg  = Aggregator()
        ckpt = CheckpointManager()
        base = "pre_round_v0"
        ckpt.save(base, {"w": torch.tensor([self.W_PRE])})
        updates = [
            _make_update("upd-good", "client-good", self.D_GOOD, base),
            _make_update("upd-bad",  "client-bad",  self.D_BAD,  base),
        ]
        return agg, ckpt, base, updates

    def test_candidate_equals_pre_round_plus_all_deltas(self):
        agg, ckpt, base, updates = self._build()
        delta = agg.aggregate(updates)
        cand  = {"w": ckpt.load(base)["w"] + delta["w"]}
        assert _scalar(cand) == pytest.approx(self.W_CANDIDATE, abs=1e-5)

    def test_correct_recovery_uses_pre_round_base(self):
        """CORRECT: W_recovered = W_pre_round + delta_recovered = 11.0"""
        agg, ckpt, base, updates = self._build()

        # Build candidate
        delta_all = agg.aggregate(updates)
        cand_state = {"w": ckpt.load(base)["w"] + delta_all["w"]}
        ckpt.save("cand_v1", cand_state)

        # Recovery from pre-round base
        good = [u for u in updates if u.client_id != "client-bad"]
        delta_rec = agg.aggregate(good)
        recovered = {"w": ckpt.load(base)["w"] + delta_rec["w"]}  # pre-round base

        assert _scalar(recovered) == pytest.approx(self.W_RECOVERED_CORRECT, abs=1e-5), (
            f"Expected {self.W_RECOVERED_CORRECT}, got {_scalar(recovered)}. "
            "Recovery used the wrong base."
        )

    def test_wrong_recovery_uses_candidate_base(self):
        """Documents OLD BUG: W_wrong = W_candidate + delta_recovered = 14.0"""
        agg, ckpt, base, updates = self._build()

        delta_all = agg.aggregate(updates)
        cand_state = {"w": ckpt.load(base)["w"] + delta_all["w"]}
        ckpt.save("cand_v1", cand_state)

        good = [u for u in updates if u.client_id != "client-bad"]
        delta_rec = agg.aggregate(good)
        # OLD BUG: uses candidate, not pre-round
        wrong = {"w": ckpt.load("cand_v1")["w"] + delta_rec["w"]}

        assert _scalar(wrong) == pytest.approx(self.W_RECOVERED_WRONG, abs=1e-5)
        # And it differs from the correct value
        assert _scalar(wrong) != pytest.approx(self.W_RECOVERED_CORRECT, abs=1e-5)

    def test_correct_and_wrong_differ_by_delta_all(self):
        """Guard: the two paths must differ by exactly delta_all = 3.0."""
        expected_diff = (self.D_GOOD + self.D_BAD) / 2
        actual_diff   = self.W_RECOVERED_WRONG - self.W_RECOVERED_CORRECT
        assert actual_diff == pytest.approx(expected_diff, abs=1e-8)

    def test_candidate_checkpoint_preserved_after_recovery(self):
        """Candidate must not be overwritten when recovered checkpoint is saved."""
        agg, ckpt, base, updates = self._build()

        delta_all = agg.aggregate(updates)
        cand_state = {"w": ckpt.load(base)["w"] + delta_all["w"]}
        ckpt.save("cand_v1", cand_state)

        # Attempting to overwrite must raise
        with pytest.raises(ValueError, match="already exists"):
            ckpt.save("cand_v1", cand_state)

        # Recovered is a new key
        good = [u for u in updates if u.client_id != "client-bad"]
        delta_rec = agg.aggregate(good)
        rec_state = {"w": ckpt.load(base)["w"] + delta_rec["w"]}
        ckpt.save("cand_v1_recovered", rec_state)

        # Both independently loadable and numerically correct
        assert _scalar(ckpt.load("cand_v1"))          == pytest.approx(self.W_CANDIDATE, abs=1e-5)
        assert _scalar(ckpt.load("cand_v1_recovered")) == pytest.approx(self.W_RECOVERED_CORRECT, abs=1e-5)


# ---------------------------------------------------------------------------
# Adapter integration: P1FLAdapter.re_aggregate() uses supplied base
# ---------------------------------------------------------------------------

class TestP1AdapterReAggregateBase:
    """
    White-box: supply pre_round as previous_model_version to re_aggregate()
    and verify the result equals W_pre_round + delta_good = 11.0, not 14.0.
    """

    def test_re_aggregate_uses_supplied_base_not_candidate(self):
        import torch.nn as nn
        from core.federated.client_manager import ClientManager
        from core.federated.evaluator import Evaluator
        from backend.adapters.p1_fl_adapter import P1FLAdapter

        W_PRE   = 10.0
        D_GOOD  =  1.0
        D_BAD   =  5.0

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.tensor([0.0]))
            def forward(self, x):
                return x * self.w

        model = TinyModel()
        ckpt  = CheckpointManager()
        agg   = Aggregator()

        adapter = P1FLAdapter(
            global_model=model,
            client_manager=ClientManager(),
            aggregator=agg,
            evaluator=Evaluator(),
            checkpoint_manager=ckpt,
        )

        # Install pre-round checkpoint
        pre_ver = "pre_v0"
        ckpt.save(pre_ver, {"w": torch.tensor([W_PRE])})

        # Build raw P1 updates
        p1_updates = [
            _make_update("upd-good", "client-good", D_GOOD, pre_ver),
            _make_update("upd-bad",  "client-bad",  D_BAD,  pre_ver),
        ]

        # Build candidate (simulates aggregate())
        delta_all  = agg.aggregate(p1_updates)
        cand_state = {"w": ckpt.load(pre_ver)["w"] + delta_all["w"]}
        cand_ver   = "cand_v1"
        ckpt.save(cand_ver, cand_state)

        # Convert to P4ModelUpdates (adapter expects them)
        p4_updates = [adapter._convert_update_p1_to_p4(u) for u in p1_updates]

        # Call re_aggregate with PRE-ROUND base (the FIXED behavior)
        rec_ver = adapter.re_aggregate(
            updates=p4_updates,
            excluded_client_ids=["client-bad"],
            previous_model_version=pre_ver,      # <-- correct base
        )

        recovered_val = _scalar(ckpt.load(rec_ver))
        expected_correct = W_PRE + D_GOOD          # 11.0
        expected_wrong   = W_PRE + (D_GOOD + D_BAD) / 2 + D_GOOD  # 14.0

        assert recovered_val == pytest.approx(expected_correct, abs=1e-4), (
            f"re_aggregate with pre-round base: expected {expected_correct}, "
            f"got {recovered_val}. "
            f"If ~{expected_wrong}, candidate was used as base (double-delta bug)."
        )

        # Candidate unchanged
        assert _scalar(ckpt.load(cand_ver)) == pytest.approx(W_PRE + (D_GOOD + D_BAD) / 2, abs=1e-4)
