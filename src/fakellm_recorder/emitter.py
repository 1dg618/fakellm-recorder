"""The emitter: turn a session into a fakellm.yaml.

The crux is choosing a `messages_contain` substring that is specific enough to
fire on the right turn but loose enough not to only match that exact transcript.
We rank candidate n-grams by inverse frequency across the whole session so shared
boilerplate (e.g. "You are a helpful assistant") is never chosen.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

from . import TARGET_FAKELLM_CONFIG_VERSION
from .conversation import _content_to_text
from .store import Exchange

Strictness = Literal["loose", "balanced", "strict"]

_WORD_RE = re.compile(r"\w+")
_STOP_PHRASES = {
    "you are a helpful assistant",
    "you are an ai assistant",
}


def _latest_user_or_tool_text(messages: list[dict[str, Any]]) -> str:
    """Text of the message that triggered this turn's response."""
    for msg in reversed(messages):
        if msg.get("role") in ("user", "tool"):
            return _content_to_text(msg.get("content"))
    return ""


def _latest_tool_result_text(messages: list[dict[str, Any]]) -> str | None:
    for msg in reversed(messages):
        if msg.get("role") == "tool":
            return _content_to_text(msg.get("content"))
    return None


def _ngrams(text: str, lo: int = 3, hi: int = 6) -> list[str]:
    words = _WORD_RE.findall(text.lower())
    grams: list[str] = []
    for n in range(lo, hi + 1):
        for i in range(len(words) - n + 1):
            grams.append(" ".join(words[i : i + n]))
    return grams


@dataclass
class _CorpusStats:
    """Document-frequency of n-grams across all triggering messages."""

    doc_freq: Counter[str] = field(default_factory=Counter)
    total_docs: int = 0

    def add(self, text: str) -> None:
        self.total_docs += 1
        for g in set(_ngrams(text)):
            self.doc_freq[g] += 1


def _pick_distinctive(text: str, stats: _CorpusStats) -> str | None:
    """Pick the rarest, longest distinctive n-gram from text."""
    candidates = set(_ngrams(text))
    if not candidates:
        return None
    scored = []
    for g in candidates:
        if g in _STOP_PHRASES:
            continue
        df = stats.doc_freq.get(g, 1)
        # prefer rare (low df) then longer (more words) then alphabetical (stable)
        scored.append((df, -len(g.split()), g))
    if not scored:
        return None
    scored.sort()
    best_df = scored[0][0]
    # Only trust it if it's reasonably distinctive across the corpus.
    if best_df > max(1, stats.total_docs // 2):
        return None
    return scored[0][2]


def _slug(text: str, maxlen: int = 24) -> str:
    words = _WORD_RE.findall(text.lower())[:4]
    s = "_".join(words) or "rule"
    return s[:maxlen]


@dataclass
class Rule:
    name: str
    when: dict[str, Any]
    respond: dict[str, Any]
    warning: str | None = None
    condition_count: int = 0


def build_rules(
    exchanges: list[Exchange], strictness: Strictness = "balanced"
) -> tuple[list[Rule], list[str]]:
    """Return (rules, warnings)."""
    stats = _CorpusStats()
    for ex in exchanges:
        stats.add(_latest_user_or_tool_text(ex.request.messages))

    rules: list[Rule] = []
    # signature -> first respond seen (for variant detection)
    seen: dict[str, str] = {}
    warnings: list[str] = []

    for ex in exchanges:
        when: dict[str, Any] = {"turn": ex.turn}

        trigger_text = _latest_user_or_tool_text(ex.request.messages)
        tool_result = _latest_tool_result_text(ex.request.messages)

        if strictness != "loose":
            # Prefer a tool-result anchor on post-tool turns; it's usually more
            # distinctive and matches how fakellm models turn-2 rules.
            if tool_result:
                anchor = _pick_distinctive(tool_result, stats) or tool_result[:40].strip()
                if anchor:
                    when["tool_result_contains"] = anchor
            else:
                sub = _pick_distinctive(trigger_text, stats)
                if sub:
                    when["messages_contain"] = sub

        if strictness == "strict":
            if ex.request.model:
                when["model_matches"] = ex.request.model
            tool_names = [
                t.get("function", {}).get("name") or t.get("name")
                for t in ex.request.tools
            ]
            tool_names = [t for t in tool_names if t]
            if tool_names:
                when["tools_include"] = tool_names[0]

        if strictness == "loose" and ex.request.model:
            when["model_matches"] = _model_glob(ex.request.model)

        respond = dict(ex.response.assembled)
        if ex.response.status != 200:
            respond["status"] = ex.response.status
            if ex.response.error:
                respond["error"] = ex.response.error

        # dedup / variant detection on the when-signature
        sig = repr(sorted(when.items()))
        respond_repr = repr(respond)
        rule = Rule(
            name=f"{ex.conversation_id[:6]}_turn{ex.turn}_{_slug(trigger_text)}",
            when=when,
            respond=respond,
            condition_count=len(when),
        )
        if sig in seen:
            if seen[sig] != respond_repr:
                rule.warning = (
                    "multiple distinct responses seen for this match; "
                    "first kept — add a header condition to disambiguate"
                )
                warnings.append(f"{rule.name}: ambiguous match (variant responses)")
            else:
                continue  # exact duplicate, collapse
        else:
            seen[sig] = respond_repr
        rules.append(rule)

    _order_and_flag_unreachable(rules, warnings)
    return rules, warnings


def _model_glob(model: str) -> str:
    """Loosen a concrete model name into a glob, e.g. gpt-4o-2024 -> gpt-4*."""
    if model.startswith("gpt-4"):
        return "gpt-4*"
    if model.startswith("gpt-3.5"):
        return "gpt-3.5*"
    if model.startswith("claude"):
        return "claude-*"
    return model


def _order_and_flag_unreachable(rules: list[Rule], warnings: list[str]) -> None:
    """First-match-wins: specific rules must precede general ones.

    Sort by (turn asc, more-conditions-first). Then flag any rule shadowed by an
    earlier rule whose conditions are a subset.
    """
    rules.sort(key=lambda r: (r.when.get("turn", 0), -r.condition_count, r.name))
    for i, rule in enumerate(rules):
        for earlier in rules[:i]:
            if _conditions_subset(earlier.when, rule.when):
                msg = f"{rule.name}: possibly unreachable (shadowed by {earlier.name})"
                if msg not in warnings:
                    warnings.append(msg)
                break


def _conditions_subset(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """True if every condition in a is also present and equal in b.

    Then a (earlier, broader-or-equal) would match anything b matches.
    """
    return all(k in b and b[k] == v for k, v in a.items())


def emit_yaml(
    exchanges: list[Exchange],
    strictness: Strictness = "balanced",
    fallback: str = "deterministic_echo",
) -> tuple[str, list[str]]:
    """Produce (yaml_text, warnings)."""
    rules, warnings = build_rules(exchanges, strictness)
    lines: list[str] = []
    lines.append(f"version: {TARGET_FAKELLM_CONFIG_VERSION}")
    lines.append("")
    lines.append("# Generated by fakellm-recorder. Review before committing.")
    lines.append(f"# strictness: {strictness}")
    lines.append("")
    lines.append("defaults:")
    lines.append(f"  fallback: {fallback}")
    lines.append("")
    lines.append("rules:")
    for rule in rules:
        if rule.warning:
            lines.append(f"  # WARNING: {rule.warning}")
        lines.append(f"  - name: {rule.name}")
        lines.append("    when:")
        for k, v in rule.when.items():
            lines.append(f"      {k}: {_yaml_scalar(v)}")
        lines.append("    respond:")
        _emit_respond(lines, rule.respond)
        lines.append("")
    return "\n".join(lines), warnings


def _emit_respond(lines: list[str], respond: dict[str, Any]) -> None:
    if not respond:
        lines.append("      {}  # deterministic echo")
        return
    if "content" in respond:
        lines.append(f"      content: {_yaml_scalar(respond['content'])}")
    if "tool_calls" in respond:
        lines.append("      tool_calls:")
        for tc in respond["tool_calls"]:
            lines.append(f"        - name: {_yaml_scalar(tc['name'])}")
            lines.append(f"          arguments: {_yaml_flow(tc['arguments'])}")
    if "status" in respond:
        lines.append(f"      status: {respond['status']}")
    if "error" in respond:
        lines.append(f"      error: {_yaml_scalar(respond['error'])}")


def _yaml_scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    # Always quote strings to stay safe with colons, leading specials, etc.
    escaped = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def _yaml_flow(obj: Any) -> str:
    """Compact JSON-ish flow mapping/sequence for arguments."""
    import json

    return json.dumps(obj, ensure_ascii=False)
