"""Lint a fakellm.yaml for common authoring mistakes.

Bonus value-add: catches unreachable rules shadowed by an earlier first-match,
unknown condition keys, and responses that set neither content nor tool_calls
unintentionally.
"""
from __future__ import annotations

from typing import Any

import yaml

KNOWN_CONDITIONS = {
    "messages_contain",
    "model_matches",
    "tools_include",
    "turn",
    "turn_in",
    "previous_message_role",
    "previous_message_contains",
    "tool_result_contains",
}
KNOWN_RESPONSE_KEYS = {"content", "tool_calls", "status", "error"}


def lint_text(text: str) -> list[str]:
    issues: list[str] = []
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as e:
        return [f"YAML parse error: {e}"]
    if not isinstance(doc, dict):
        return ["top-level document is not a mapping"]

    rules = doc.get("rules") or []
    if not isinstance(rules, list):
        return ["`rules:` is not a list"]

    seen_signatures: list[dict[str, Any]] = []
    for i, rule in enumerate(rules):
        name = rule.get("name", f"<rule #{i}>")
        when = rule.get("when") or {}
        respond = rule.get("respond")

        for key in when:
            if key.startswith("header."):
                continue
            if key not in KNOWN_CONDITIONS:
                issues.append(f"{name}: unknown condition key '{key}'")

        if respond is None:
            issues.append(f"{name}: missing `respond:` block")
        elif isinstance(respond, dict):
            for key in respond:
                if key not in KNOWN_RESPONSE_KEYS:
                    issues.append(f"{name}: unknown respond key '{key}'")
            has_status_err = "status" in respond and respond.get("status", 200) >= 400
            if not respond.get("content") and not respond.get("tool_calls") and not has_status_err:
                if respond != {}:
                    issues.append(
                        f"{name}: respond sets neither content nor tool_calls "
                        "(will fall back to echo — intentional?)"
                    )

        # unreachable: an earlier rule whose conditions are a subset of this one
        for earlier in seen_signatures:
            if all(k in when and when[k] == v for k, v in earlier["when"].items()):
                issues.append(
                    f"{name}: possibly unreachable (shadowed by {earlier['name']})"
                )
                break
        seen_signatures.append({"name": name, "when": when})

    return issues
