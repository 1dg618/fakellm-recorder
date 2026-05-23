"""Scrubbing of credentials and PII.

Recordings become commitable artifacts, so scrubbing is on by default. Header
capture is allowlist-based (never persist auth); body scrubbing is pattern-based
with user-supplied regex extensions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Pattern

# Only these request headers are ever persisted. Everything else (notably
# Authorization, x-api-key, anthropic auth headers) is dropped at capture time.
HEADER_ALLOWLIST = {
    "x-test-scenario",
    "x-fakellm-conversation-id",
    "content-type",
}

REDACTION = "<redacted>"

# Built-in patterns. Conservative on purpose — better to miss an exotic token
# than to mangle legitimate prompt text. Users add their own via custom regexes.
_BUILTIN_PATTERNS: list[tuple[str, Pattern[str]]] = [
    ("email", re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")),
    ("openai_key", re.compile(r"sk-[A-Za-z0-9_-]{16,}")),
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_-]{16,}")),
    ("bearer", re.compile(r"Bearer\s+[A-Za-z0-9._-]{12,}")),
]


@dataclass
class Scrubber:
    enabled: bool = True
    custom_patterns: list[Pattern[str]] = field(default_factory=list)
    # name -> count of redactions, for the summary report
    counts: dict[str, int] = field(default_factory=dict)

    def filter_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Allowlist headers regardless of whether scrubbing is enabled.

        Credentials must never be persisted even if a user disables PII scrubbing.
        """
        return {
            k.lower(): v
            for k, v in headers.items()
            if k.lower() in HEADER_ALLOWLIST
        }

    def scrub_text(self, text: str) -> str:
        if not self.enabled or not text:
            return text
        result = text
        for name, pat in _BUILTIN_PATTERNS:
            result, n = pat.subn(REDACTION, result)
            if n:
                self.counts[name] = self.counts.get(name, 0) + n
        for i, pat in enumerate(self.custom_patterns):
            result, n = pat.subn(REDACTION, result)
            if n:
                key = f"custom[{i}]"
                self.counts[key] = self.counts.get(key, 0) + n
        return result

    def scrub_obj(self, obj: Any) -> Any:
        """Recursively scrub all strings inside a JSON-like structure."""
        if not self.enabled:
            return obj
        if isinstance(obj, str):
            return self.scrub_text(obj)
        if isinstance(obj, list):
            return [self.scrub_obj(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self.scrub_obj(v) for k, v in obj.items()}
        return obj

    def report(self) -> str:
        if not self.counts:
            return "Scrubber: nothing redacted."
        lines = ["Scrubber redactions:"]
        for name, count in sorted(self.counts.items()):
            lines.append(f"  {name}: {count}")
        return "\n".join(lines)
