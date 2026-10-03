"""Webhook alerts (Slack-compatible) when a run fails with a high failure probability."""
from __future__ import annotations

import json
import logging
import urllib.request

from .config import settings

logger = logging.getLogger(__name__)


def alert_message(run: dict, diagnosis: dict, dashboard_url: str = "http://localhost:8010") -> dict:
    root = diagnosis.get("root_cause") or {}
    text = (f":rotating_light: Black Box: run `{run['run_id']}` failed "
            f"(p_fail {diagnosis.get('p_fail', 0):.0%}). Root cause: step {root.get('step')} "
            f"`{root.get('node')}` ({root.get('confidence', 0):.0%} blame). "
            f"{diagnosis.get('evidence', {}).get('summary', '')}\nTask: {run.get('prompt', '')[:200]}\n{dashboard_url}")
    return {"text": text, "run_id": run["run_id"], "p_fail": diagnosis.get("p_fail"),
            "root_cause": root, "task": run.get("prompt")}


def maybe_alert(run: dict, diagnosis: dict) -> bool:
    """POST to ALERT_WEBHOOK_URL when the run failed and p_fail >= ALERT_P_FAIL_THRESHOLD."""
    cfg = settings()
    if not cfg.alert_webhook or run.get("status") != "FAILED" or (diagnosis.get("p_fail") or 0) < cfg.alert_threshold:
        return False
    body = json.dumps(alert_message(run, diagnosis)).encode()
    request = urllib.request.Request(cfg.alert_webhook, data=body, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(request, timeout=5).close()
        return True
    except Exception as exc:  # An alert must never break recording.
        logger.warning("Alert webhook failed: %s", exc)
        return False
