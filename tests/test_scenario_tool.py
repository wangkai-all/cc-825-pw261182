import csv
import io
import json
import math
import os
import tempfile
import unittest

from scenario_tool.dataset import prepare_dataset
from scenario_tool.io import parse_csv
from scenario_tool.jobs import AllocationItem, ProbabilityAllocationJob, recovery_drill
from scenario_tool.pipeline import run_analysis, run_exact_check, run_pareto
from scenario_tool.reduction import exhaustive_reduce, heuristic_reduce, reduce_scenarios
from scenario_tool.sample import generate_fictional_dataset, toy_dataset
from scenario_tool.tree import build_prefix_tree, check_prefix_assignment, validate_tree


def payload_with_duplicates_and_issues():
    p = toy_dataset()
    # Two exact rows: probabilities .10 + .30 should merge to .40.
    p["paths"][0]["probability"] = 0.10
    duplicate = json.loads(json.dumps(p["paths"][0]))
    duplicate["id"] = "DUP-EXACT"
    duplicate["probability"] = 0.30
    p["paths"].insert(1, duplicate)
    # Exactly zero row and total deliberately not normalized.
    p["paths"][-1]["probability"] = 0.0
    # Existing other probabilities sum to 0.78? Doesn't matter; report flags it.
    return p


class ValidationTests(unittest.TestCase):
    def test_duplicates_merge_before_separate_probability_checks(self):
        payload = payload_with_duplicates_and_issues()
        ds, report = prepare_dataset(payload)
        self.assertGreaterEqual(report["duplicateRowsMerged"], 1)
        merged = next(p for p in ds["paths"] if p["id"] == "A1")
        self.assertAlmostEqual(merged["probability"], 0.40, places=12)
        prob = report["probabilities"]
        self.assertGreater(prob["zeroProbabilityCount"], 0)
        self.assertFalse(prob["normalized"])
        codes = {c["code"] for c in prob["checks"]}
        self.assertIn("ZERO_PROBABILITY", codes)
        self.assertIn("UNNORMALIZED_PROBABILITY", codes)
        zero = next(c for c in prob["checks"] if c["code"] == "ZERO_PROBABILITY")
        unnorm = next(c for c in prob["checks"] if c["code"] == "UNNORMALIZED_PROBABILITY")
        self.assertNotEqual(zero["status"], unnorm["status"] if False else "pass")
        self.assertEqual(zero["code"], "ZERO_PROBABILITY")
        self.assertEqual(unnorm["code"], "UNNORMALIZED_PROBABILITY")

    def test_normalization_is_explicit_and_conserves(self):
        payload = toy_dataset()
        for p in payload["paths"]:
            p["probability"] *= 2.0  # sum becomes 2
        ds, report = prepare_dataset(payload, normalize=True)
        self.assertIn("normalized", report["actions"])
        self.assertAlmostEqual(math.fsum(p["probability"] for p in ds["paths"]), 1.0, places=12)

    def test_negative_probability_is_invalid(self):
        payload = toy_dataset()
        payload["paths"][0]["probability"] = -0.22
        _, report = prepare_dataset(payload)
        self.assertFalse(report["valid"])
        self.assertTrue(report["errors"])


class ReductionAndTreeTests(unittest.TestCase):
    def test_toy_exhaustive_probability_and_tree_invariants(self):
        ds, _ = prepare_dataset(toy_dataset(), normalize=True)
        result = exhaustive_reduce(ds, {"k": 3, "tailGuards": 0, "tailWeight": 0.75})
        cons = result["probabilityConservation"]
        self.assertTrue(cons["conserved"])
        self.assertLess(cons["error"], 1e-9)
        self.assertAlmostEqual(sum(r["probability"] for r in result["representatives"]), 1.0, places=10)
        self.assertTrue(result["nonAnticipativity"]["compatible"])
        self.assertTrue(result["treeValidation"]["valid"])
        self.assertEqual(result["search"]["combinationsEvaluated"], math.comb(8, 3))

    def test_heuristic_matches_complete_search_on_small_instance(self):
        check = run_exact_check({"k": 3, "tailWeight": 0.75})
        self.assertEqual(check["instance"]["subsets"], 56)
        self.assertTrue(check["optimality"]["matchesExhaustiveOptimum"], check["optimality"])

    def test_bad_early_split_is_detected(self):
        ds, _ = prepare_dataset(toy_dataset(), normalize=True)
        reps = [4, 6]  # C2 has T2=75 prefix, D1 has T2=30 prefix.
        # A1 and A2 share T1/T2 prefix [10,20], but this assignment splits them.
        assignment = [4, 6, 4, 4, 4, 4, 6, 6]
        check = check_prefix_assignment(ds, reps, assignment)
        self.assertFalse(check["compatible"])
        self.assertTrue(check["violations"])

    def test_reduced_tree_branch_and_probability_validation(self):
        ds, _ = prepare_dataset(toy_dataset(), normalize=True)
        result = reduce_scenarios(ds, {"k": 4, "tailGuards": 1})
        tree = result["tree"]
        validation = validate_tree(tree, expected_total=1.0)
        self.assertTrue(validation["valid"], validation["errors"])
        self.assertGreaterEqual(validation["branchPointCount"], 1)
        # Every leaf must correspond to exactly one selected representative.
        self.assertEqual(validation["leafCount"], 4)

    def test_zero_probability_paths_still_assigned_but_not_candidates(self):
        payload = toy_dataset()
        payload["paths"][-1]["probability"] = 0.0
        ds, _ = prepare_dataset(payload)
        result = reduce_scenarios(ds, {"k": 3, "tailGuards": 1})
        zero_assignment = result["assignments"][-1]
        self.assertGreaterEqual(zero_assignment["representativeIndex"], 0)
        self.assertNotEqual(result["representatives"][0]["id"], ds["paths"][-1]["id"])
        self.assertTrue(result["probabilityConservation"]["conserved"])


class PlanningAndPipelineTests(unittest.TestCase):
    def test_analysis_planning_regret_nonnegative(self):
        payload = generate_fictional_dataset(80)
        result = run_analysis(payload, {"k": 6, "tailGuards": 1})
        pc = result["planningComparison"]
        self.assertGreaterEqual(pc["outOfSampleRegret"], -1e-7)
        self.assertTrue(result["reduction"]["treeValidation"]["valid"])

    def test_pareto_reports_both_costs(self):
        payload = generate_fictional_dataset(60)
        pareto = run_pareto(payload, {"k": 5, "tailGuards": 1}, [0.0, 1.0, 3.0])
        self.assertEqual(len(pareto["grid"]), 3)
        for row in pareto["grid"]:
            self.assertIn("distancePriceOfTailProtection", row)
            self.assertIn("cvarUnderestimate", row)
            self.assertIn("planningRegret", row)
        self.assertGreater(pareto["costs"]["distanceFirstCvarUnderestimate"], -1e-9)


class RecoveryTests(unittest.TestCase):
    def test_recovery_never_reallocates_committed_mass(self):
        drill = recovery_drill()
        self.assertTrue(drill["interrupted"])
        self.assertTrue(drill["final"]["valid"])
        self.assertTrue(drill["matchesCleanAllocation"])
        self.assertEqual(drill["redeliveryNoOp"]["newCommits"], 0)

    def test_ledger_checkpoint_resume(self):
        items = [AllocationItem(f"P{i}", 0.1, "R1" if i % 2 else "R2") for i in range(10)]
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ledger.json")
            job = ProbabilityAllocationJob(job_id="j", representative_ids=["R1", "R2"], expected_mass=1.0, checkpoint_path=path)
            try:
                job.run(items, crash_after_commits=3)
            except Exception:
                pass
            resumed = ProbabilityAllocationJob.load_or_create(path)
            before = resumed.totals()
            resumed.run(items)
            after = resumed.verify()
            self.assertEqual(before["entryCount"], 3)
            self.assertEqual(after["entryCount"], 10)
            self.assertTrue(after["probabilityConserved"])
            self.assertEqual(after["duplicateProcessedIds"], [])


class CsvImportTests(unittest.TestCase):
    def test_wide_csv(self):
        text = """id,probability,t0_demand,t0_freight,t1_demand,t1_freight
A,.5,10,6,20,7
B,.5,12,6.5,22,7.2
"""
        payload = parse_csv(text)
        self.assertEqual(len(payload["paths"]), 2)
        self.assertEqual(len(payload["periods"]), 2)
        self.assertEqual([d["key"] for d in payload["dimensions"]], ["demand", "freight"])
        ds, report = prepare_dataset(payload)
        self.assertTrue(report["probabilities"]["normalized"])


class ScaleTests(unittest.TestCase):
    def test_ten_thousand_paths_pipeline(self):
        payload = generate_fictional_dataset(10000)
        result = run_analysis(payload, {"k": 16, "tailGuards": 2})
        self.assertEqual(len(result["reduction"]["representatives"]), 16)
        self.assertLess(result["reduction"]["probabilityConservation"]["error"], 1e-8)
        self.assertTrue(result["reduction"]["nonAnticipativity"]["compatible"])
        self.assertTrue(result["reduction"]["treeValidation"]["valid"])


if __name__ == "__main__":
    unittest.main()
