"""Unit tests for the claim payout tool (no OpenAI calls)."""
import re

from app.tools.claims_tools import estimate_payout_tool, generate_claim_id


def run_payout(repair_cost, plan="Comprehensive"):
    """Call the estimate_payout tool the way the agent does and return its result."""
    return estimate_payout_tool.invoke({
        "estimated_repair_cost_inr": repair_cost,
        "policy_idv": 850_000,
        "plan": plan,
    })


class TestEstimatePayout:
    """Payout maths, claim ID and surveyor rule."""

    def test_comprehensive_applies_depreciation_and_deductible(self):
        """₹1,50,000 repair → 30% depreciation (₹45,000) and ₹2,500 deductible."""
        result = run_payout(150_000)
        assert result["depreciation_deduction_inr"] == 45_000
        assert result["estimated_payout_inr"] == 102_500
        assert result["out_of_pocket_inr"] == 47_500

    def test_zero_dep_skips_depreciation(self):
        """Zero-Dep plans only lose the deductible."""
        result = run_payout(150_000, plan="Comprehensive + Zero Depreciation")
        assert result["estimated_payout_inr"] == 147_500

    def test_surveyor_needed_only_above_threshold(self):
        """A surveyor visit is flagged only when the payout exceeds ₹50,000."""
        assert run_payout(150_000)["surveyor_visit_required"] is True
        assert run_payout(40_000)["surveyor_visit_required"] is False

    def test_every_payout_has_a_claim_id(self):
        """The summary always has a claim reference to show the customer."""
        assert re.fullmatch(r"CLM-\d{4}-\d{5}", run_payout(20_000)["claim_id"])


def test_claim_ids_are_well_formed():
    """generate_claim_id returns CLM-YYYY-NNNNN."""
    for _ in range(50):
        assert re.fullmatch(r"CLM-\d{4}-\d{5}", generate_claim_id())
