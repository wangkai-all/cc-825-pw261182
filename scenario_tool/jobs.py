"""Idempotent probability-mass allocation jobs with crash recovery.

A reduction job moves each original path's probability mass exactly once into
one representative bucket.  The durable ledger records processed path IDs and
committed representative masses.  When a task is recovered, already-processed
IDs are skipped and their committed mass is never redistributed.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


class JobInterrupted(RuntimeError):
    """Raised by deterministic crash-injection in tests/drills."""


@dataclass
class AllocationItem:
    path_id: str
    mass: float
    representative_id: str


class ProbabilityAllocationJob:
    SCHEMA = "scenario-tool-mass-ledger/v1"

    def __init__(
        self,
        *,
        job_id: str,
        representative_ids: List[str],
        expected_mass: float,
        checkpoint_path: Optional[str] = None,
    ):
        self.job_id = job_id
        self.representative_ids = list(representative_ids)
        self.expected_mass = float(expected_mass)
        self.checkpoint_path = checkpoint_path
        self.entries: List[Dict[str, Any]] = []
        self.processed: Dict[str, Dict[str, Any]] = {}
        self.mass_by_representative: Dict[str, float] = {rid: 0.0 for rid in representative_ids}
        self.status = "new"
        self.allocations_attempted = 0

    @classmethod
    def load_or_create(cls, checkpoint_path: str) -> "ProbabilityAllocationJob":
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            job = cls(
                job_id=raw["jobId"],
                representative_ids=raw["representativeIds"],
                expected_mass=raw["expectedMass"],
                checkpoint_path=checkpoint_path,
            )
            job.entries = raw.get("entries", [])
            job.processed = {e["pathId"]: e for e in job.entries}
            job.mass_by_representative = dict(raw.get("massByRepresentative", {}))
            for rid in job.representative_ids:
                job.mass_by_representative.setdefault(rid, 0.0)
            job.status = raw.get("status", "recovered")
            return job
        job = cls(job_id="recovery-drill", representative_ids=[], expected_mass=0.0, checkpoint_path=checkpoint_path)
        return job

    def _save(self) -> None:
        if not self.checkpoint_path:
            return
        payload = {
            "schema": self.SCHEMA,
            "jobId": self.job_id,
            "representativeIds": self.representative_ids,
            "expectedMass": self.expected_mass,
            "status": self.status,
            "entries": self.entries,
            "massByRepresentative": self.mass_by_representative,
        }
        directory = os.path.dirname(os.path.abspath(self.checkpoint_path))
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".ledger-", suffix=".json", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, self.checkpoint_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def commit(self, item: AllocationItem) -> bool:
        """Commit one item. Returns False if it was already processed."""
        if item.path_id in self.processed:
            # Recovery/idempotent redelivery: never move processed mass again.
            return False
        if item.representative_id not in self.mass_by_representative:
            raise KeyError(f"unknown representative {item.representative_id}")

        entry = {
            "pathId": item.path_id,
            "mass": float(item.mass),
            "representativeId": item.representative_id,
            "sequence": len(self.entries),
        }
        self.entries.append(entry)
        self.processed[item.path_id] = entry
        self.mass_by_representative[item.representative_id] += float(item.mass)
        self.status = "running"
        self._save()
        return True

    def run(
        self,
        items: List[AllocationItem],
        *,
        crash_after_commits: Optional[int] = None,
    ) -> Dict[str, Any]:
        self.status = "running"
        self._save()
        initial_entries = len(self.entries)
        commits_this_run = 0
        for item in items:
            committed = self.commit(item)
            if committed:
                commits_this_run += 1
                if crash_after_commits is not None and commits_this_run >= crash_after_commits:
                    self.status = "interrupted"
                    self._save()
                    raise JobInterrupted(
                        f"injected crash after {commits_this_run} new commits; "
                        f"{len(self.entries)} total entries durable"
                    )
        self.status = "completed"
        self._save()
        return {
            "entriesBeforeRun": initial_entries,
            "newCommits": commits_this_run,
            "totalEntries": len(self.entries),
            "status": self.status,
        }

    def totals(self) -> Dict[str, Any]:
        total = sum(e["mass"] for e in self.entries)
        return {
            "processedPathCount": len(self.processed),
            "entryCount": len(self.entries),
            "allocatedMass": total,
            "expectedMass": self.expected_mass,
            "conservationError": abs(total - self.expected_mass),
            "massByRepresentative": dict(self.mass_by_representative),
            "status": self.status,
        }

    def verify(self, tol: float = 1e-9) -> Dict[str, Any]:
        seen = [e["pathId"] for e in self.entries]
        duplicate_ids = sorted({pid for pid in seen if seen.count(pid) > 1})
        totals = self.totals()
        conserved = totals["conservationError"] <= tol * max(1.0, abs(self.expected_mass))
        return {
            "valid": not duplicate_ids and conserved,
            "duplicateProcessedIds": duplicate_ids,
            "probabilityConserved": conserved,
            **totals,
        }


def recovery_drill(items: Optional[List[AllocationItem]] = None) -> Dict[str, Any]:
    """Crash after a partial commit, resume, and prove no mass is reallocated."""
    if items is None:
        rep_ids = ["R-A", "R-B", "R-C"]
        items = [
            AllocationItem(f"S{i:03d}", 0.01 * (i % 7 + 1), rep_ids[i % len(rep_ids)])
            for i in range(12)
        ]
    else:
        rep_ids = sorted({x.representative_id for x in items})

    expected_mass = sum(x.mass for x in items)
    with tempfile.TemporaryDirectory(prefix="scenario-recovery-") as tmp:
        checkpoint = os.path.join(tmp, "ledger.json")
        job = ProbabilityAllocationJob(
            job_id="deterministic-recovery-drill",
            representative_ids=rep_ids,
            expected_mass=expected_mass,
            checkpoint_path=checkpoint,
        )
        crash_after = max(1, len(items) // 3)
        interrupted = False
        try:
            job.run(items, crash_after_commits=crash_after)
        except JobInterrupted:
            interrupted = True

        partial = job.totals()
        resumed = ProbabilityAllocationJob.load_or_create(checkpoint)
        before_resume = resumed.totals()
        resume_report = resumed.run(items)  # redeliver full batch, including old IDs
        after_resume = resumed.verify()

        # Redelivering again after completion must be a no-op.
        noop_report = resumed.run(items)
        final_verify = resumed.verify()

    clean_mass: Dict[str, float] = {rid: 0.0 for rid in rep_ids}
    for item in items:
        clean_mass[item.representative_id] += item.mass

    return {
        "interrupted": interrupted,
        "crashAfterNewCommits": crash_after,
        "partial": partial,
        "loadedFromCheckpoint": before_resume,
        "resume": resume_report,
        "afterResume": after_resume,
        "redeliveryNoOp": noop_report,
        "final": final_verify,
        "cleanMassByRepresentative": clean_mass,
        "matchesCleanAllocation": all(
            abs(final_verify["massByRepresentative"].get(rid, 0.0) - clean_mass[rid]) < 1e-12
            for rid in rep_ids
        ),
        "rule": "recovered tasks skip processed path IDs and never redistribute committed mass",
    }
