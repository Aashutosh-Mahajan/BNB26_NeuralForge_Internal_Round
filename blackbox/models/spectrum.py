"""Ochiai component suspiciousness from a historical labelled population."""
import math


def ochiai(runs):
    failed = sum(run.get("success") is False for run in runs)
    counts = {}
    for run in runs:
        if run.get("success") not in (True, False):
            continue
        for name in {step.get("node_name", "unknown") for step in run.get("steps", [])}:
            record = counts.setdefault(name, {"failed": 0, "passed": 0})
            record["passed" if run["success"] else "failed"] += 1
    return [{"component": name, **count,
             "suspiciousness": count["failed"] / math.sqrt(failed * (count["failed"] + count["passed"]))
             if failed and count["failed"] + count["passed"] else 0.0}
            for name, count in sorted(counts.items())]
