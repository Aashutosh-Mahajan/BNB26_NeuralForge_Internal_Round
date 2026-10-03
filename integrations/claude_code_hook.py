"""Claude Code hook -> Black Box. Records each tool call of a Claude Code session.

Simplest: copy integrations/claude_code_settings.json (built-in "http" hooks, no script).
Alternative command hooks:
  PostToolUse, PostToolUseFailure, UserPromptSubmit and Stop run:
      python integrations/claude_code_hook.py
The hook never blocks Claude Code: it always exits 0, even if Black Box is offline.
Capability: record + diagnose only (no checkpoint, fork or resume of a coding session).
"""
import json
import os
import sys
import urllib.request

URL = os.environ.get("BLACKBOX_URL", "http://127.0.0.1:8010").rstrip("/") + "/api/ingest/claude-code"


def main():
    try:
        event = json.load(sys.stdin)
        request = urllib.request.Request(URL, data=json.dumps(event).encode(), headers={"Content-Type": "application/json"})
        urllib.request.urlopen(request, timeout=3).close()
    except Exception:
        pass  # observation must never interfere with the coding session
    sys.exit(0)


if __name__ == "__main__":
    main()
