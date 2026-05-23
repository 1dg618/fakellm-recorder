"""Unit tests for the pure components."""
from __future__ import annotations

import re

from fakellm_recorder.conversation import conversation_id, first_user_text
from fakellm_recorder.emitter import build_rules, emit_yaml
from fakellm_recorder.linter import lint_text
from fakellm_recorder.scrub import Scrubber
from fakellm_recorder.store import Exchange, RequestRecord, ResponseRecord
from fakellm_recorder.streaming import assemble_anthropic, assemble_openai


# ---- conversation bucketing ----

def test_conversation_id_header_wins():
    msgs = [{"role": "user", "content": "hello"}]
    assert conversation_id(msgs, "explicit-id") == "explicit-id"


def test_conversation_id_stable_across_turns():
    first = [{"role": "user", "content": "research fakellm please"}]
    later = first + [
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "now summarize"},
    ]
    # id is keyed on the first user message, so it must not change as turns grow
    assert conversation_id(first, None) == conversation_id(later, None)


def test_first_user_text_handles_block_content():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "blocks work"}]}]
    assert first_user_text(msgs) == "blocks work"


# ---- streaming assembly ----

def test_assemble_openai_content_and_tool_calls():
    raw = (
        'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":'
        '{"name":"web_search","arguments":"{\\"q\\":"}}]}}]}\n\n'
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":'
        '{"arguments":"\\"fakellm\\"}"}}]}}]}\n\n'
        "data: [DONE]\n\n"
    )
    ar = assemble_openai(raw)
    assert ar.content == "Hello"
    assert ar.tool_calls == [{"name": "web_search", "arguments": {"q": "fakellm"}}]


def test_assemble_anthropic_content_and_tool_use():
    raw = (
        'event: content_block_start\n'
        'data: {"type":"content_block_start","index":0,'
        '"content_block":{"type":"text","text":""}}\n\n'
        'event: content_block_delta\n'
        'data: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"text_delta","text":"Hi there"}}\n\n'
        'event: content_block_start\n'
        'data: {"type":"content_block_start","index":1,'
        '"content_block":{"type":"tool_use","name":"get_weather"}}\n\n'
        'event: content_block_delta\n'
        'data: {"type":"content_block_delta","index":1,'
        '"delta":{"type":"input_json_delta","partial_json":"{\\"city\\":\\"NYC\\"}"}}\n\n'
        'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )
    ar = assemble_anthropic(raw)
    assert ar.content == "Hi there"
    assert ar.tool_calls == [{"name": "get_weather", "arguments": {"city": "NYC"}}]


# ---- scrubber ----

def test_scrubber_strips_credentials_via_allowlist():
    sc = Scrubber()
    headers = {"Authorization": "Bearer secret", "x-test-scenario": "happy", "X-Api-Key": "sk-abc"}
    filtered = sc.filter_headers(headers)
    assert "authorization" not in filtered
    assert "x-api-key" not in filtered
    assert filtered["x-test-scenario"] == "happy"


def test_scrubber_redacts_email_and_keys():
    sc = Scrubber()
    text = "contact me at jane@example.com with key sk-abc123def456ghi789"
    out = sc.scrub_text(text)
    assert "jane@example.com" not in out
    assert "sk-abc123def456ghi789" not in out
    assert sc.counts.get("email") == 1


def test_scrubber_custom_pattern():
    sc = Scrubber(custom_patterns=[re.compile(r"ACCT-\d+")])
    out = sc.scrub_text("account ACCT-99887")
    assert "ACCT-99887" not in out


# ---- emitter ----

def _ex(conv, turn, msgs, content=None, tool_calls=None, status=200, model="gpt-4o", tools=None):
    assembled = {}
    if content:
        assembled["content"] = content
    if tool_calls:
        assembled["tool_calls"] = tool_calls
    return Exchange(
        conversation_id=conv,
        turn=turn,
        api="openai",
        request=RequestRecord(model=model, messages=msgs, tools=tools or [], stream=False),
        response=ResponseRecord(status=status, assembled=assembled),
    )


def test_emitter_multiturn_agent_flow():
    exchanges = [
        _ex("conv1", 1,
            [{"role": "system", "content": "You are a helpful assistant"},
             {"role": "user", "content": "Please research the fakellm library"}],
            tool_calls=[{"name": "web_search", "arguments": {"query": "fakellm"}}]),
        _ex("conv1", 2,
            [{"role": "user", "content": "Please research the fakellm library"},
             {"role": "assistant", "content": ""},
             {"role": "tool", "content": "found 3 relevant results"}],
            content="Based on the search, I found what you were looking for."),
    ]
    rules, warnings = build_rules(exchanges, "balanced")
    assert len(rules) == 2
    # turn 1 rule fires on turn 1; boilerplate must not be the chosen substring
    t1 = next(r for r in rules if r.when["turn"] == 1)
    assert "you are a helpful assistant" not in str(t1.when).lower()
    assert t1.respond["tool_calls"][0]["name"] == "web_search"
    # turn 2 prefers a tool_result anchor
    t2 = next(r for r in rules if r.when["turn"] == 2)
    assert "tool_result_contains" in t2.when


def test_emitter_dedups_identical():
    msgs = [{"role": "user", "content": "ping unique-xyz"}]
    exchanges = [
        _ex("c", 1, msgs, content="pong"),
        _ex("c", 1, msgs, content="pong"),
    ]
    rules, _ = build_rules(exchanges, "balanced")
    assert len(rules) == 1


def test_emitter_flags_variant_responses():
    msgs = [{"role": "user", "content": "ping ambiguous-token"}]
    exchanges = [
        _ex("c", 1, msgs, content="pong A"),
        _ex("c", 1, msgs, content="pong B"),
    ]
    rules, warnings = build_rules(exchanges, "balanced")
    assert any("ambiguous" in w for w in warnings)


def test_emit_yaml_is_lint_clean():
    exchanges = [
        _ex("conv1", 1,
            [{"role": "user", "content": "translate this distinctive phrase to french"}],
            content="bonjour"),
    ]
    text, _ = emit_yaml(exchanges, "balanced")
    assert "version: 1" in text
    assert lint_text(text) == []


# ---- linter ----

def test_linter_detects_unknown_condition():
    bad = """
version: 1
rules:
  - name: r1
    when:
      not_a_real_key: foo
    respond:
      content: "hi"
"""
    issues = lint_text(bad)
    assert any("unknown condition" in i for i in issues)


def test_linter_detects_unreachable():
    bad = """
version: 1
rules:
  - name: broad
    when:
      turn: 1
    respond:
      content: "first"
  - name: narrow
    when:
      turn: 1
      messages_contain: "specific"
    respond:
      content: "second"
"""
    issues = lint_text(bad)
    assert any("unreachable" in i for i in issues)
