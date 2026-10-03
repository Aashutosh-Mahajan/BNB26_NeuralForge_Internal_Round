"""Needleman-Wunsch alignment and dependency-free recursive JSON diffs."""
from __future__ import annotations
from typing import Any


def json_changes(before: Any, after: Any, path: str = "$", limit: int = 100) -> list[dict]:
    """Return typed leaf changes; dictionary order never creates a difference."""
    changes: list[dict] = []

    def visit(a: Any, b: Any, current: str) -> None:
        if len(changes) >= limit:
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(a.keys() | b.keys(), key=str):
                child = f"{current}.{key}"
                if key not in a:
                    changes.append({"path": child, "kind": "added", "before": None, "after": b[key]})
                elif key not in b:
                    changes.append({"path": child, "kind": "removed", "before": a[key], "after": None})
                else:
                    visit(a[key], b[key], child)
                if len(changes) >= limit:
                    break
        elif isinstance(a, list) and isinstance(b, list):
            for i in range(max(len(a), len(b))):
                child = f"{current}[{i}]"
                if i >= len(a):
                    changes.append({"path": child, "kind": "added", "before": None, "after": b[i]})
                elif i >= len(b):
                    changes.append({"path": child, "kind": "removed", "before": a[i], "after": None})
                else:
                    visit(a[i], b[i], child)
                if len(changes) >= limit:
                    break
        elif type(a) is not type(b) or a != b:
            changes.append({"path": current, "kind": "changed", "before": a, "after": b})

    visit(before, after, path)
    return changes[:limit]


def _node(step: dict) -> str:
    return str(step.get("node_name", step.get("node", "unknown")))


def _align(a: list[dict], b: list[dict]) -> list[tuple[dict | None, dict | None]]:
    """Global alignment: exact node +3, same type +0, mismatch -2, gap -2."""
    n, m = len(a), len(b)
    score = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        score[i][0] = -2 * i
    for j in range(m + 1):
        score[0][j] = -2 * j

    def match(x: dict, y: dict) -> int:
        if _node(x) == _node(y):
            return 3
        return 0 if x.get("node_type") and x.get("node_type") == y.get("node_type") else -2

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score[i][j] = max(score[i - 1][j - 1] + match(a[i - 1], b[j - 1]),
                              score[i - 1][j] - 2, score[i][j - 1] - 2)
    aligned = []
    i, j = n, m
    while i or j:
        if i and j and score[i][j] == score[i - 1][j - 1] + match(a[i - 1], b[j - 1]):
            aligned.append((a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i and score[i][j] == score[i - 1][j] - 2:
            aligned.append((a[i - 1], None))
            i -= 1
        else:
            aligned.append((None, b[j - 1]))
            j -= 1
    return list(reversed(aligned))


def compare_runs(a: dict, b: dict) -> dict:
    """Compare observable execution/state. Outcomes are reporting metadata only."""
    aligned_steps, changes = [], []
    first_divergence = None
    for left, right in _align(a.get("steps", []), b.get("steps", [])):
        left_id = left.get("step_id") if left else None
        right_id = right.get("step_id") if right else None
        delta = []
        if left is None or right is None:
            status = "added" if left is None else "removed"
            delta = [{"path": "$", "kind": status, "before": left, "after": right}]
        else:
            for field in ("node_name", "input", "output", "state_after", "tool_error"):
                delta.extend(json_changes(left.get(field), right.get(field), f"$.{field}"))
            status = "changed" if delta else "unchanged"
        if delta and first_divergence is None:
            first_divergence = left_id if left_id is not None else right_id
        entry = {"a_step": left_id, "b_step": right_id, "node": _node(left or right or {}),
                 "status": status, "changes": delta}
        aligned_steps.append(entry)
        if delta:
            changes.append(entry)
    return {"a": a.get("run_id"), "b": b.get("run_id"), "first_divergence": first_divergence,
            "changes": changes, "aligned_steps": aligned_steps,
            "outcome": {"before": a.get("success"), "after": b.get("success"),
                        "changed": a.get("success") != b.get("success"),
                        "answer_before": a.get("final_answer"), "answer_after": b.get("final_answer")}}
