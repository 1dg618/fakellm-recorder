"""End-to-end: drive the proxy against a stubbed upstream, then emit + lint.

We monkeypatch httpx.AsyncClient inside the proxy so no real network call is made.
This proves the full loop: request -> capture -> store -> emit -> lint.
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from starlette.testclient import TestClient

from fakellm_recorder import proxy as proxy_mod
from fakellm_recorder.emitter import emit_yaml
from fakellm_recorder.linter import lint_text
from fakellm_recorder.proxy import build_app
from fakellm_recorder.scrub import Scrubber
from fakellm_recorder.store import SessionStore


class _StubResponse:
    def __init__(self, status_code, text, headers):
        self.status_code = status_code
        self.text = text
        self.content = text.encode("utf-8")
        self.headers = headers


class _StubClient:
    """Stands in for httpx.AsyncClient; returns canned upstream responses."""

    _queue: list[_StubResponse] = []

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def request(self, method, url, **kwargs):
        return _StubClient._queue.pop(0)


@pytest.fixture
def patched_proxy(monkeypatch, tmp_path):
    monkeypatch.setattr(proxy_mod.httpx, "AsyncClient", _StubClient)
    store = SessionStore(tmp_path / "session.jsonl")
    app = build_app(store, Scrubber(enabled=True), "auto")
    client = TestClient(app)
    return client, store, tmp_path


def test_e2e_openai_nonstreaming(patched_proxy):
    client, store, tmp_path = patched_proxy
    # Stub upstream: a normal OpenAI chat completion
    _StubClient._queue = [
        _StubResponse(
            200,
            json.dumps({
                "choices": [{"message": {"role": "assistant",
                                         "content": "The capital is Paris."}}]
            }),
            {"content-type": "application/json"},
        )
    ]
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [
            {"role": "user", "content": "What is the capital of France precisely"}
        ]},
        headers={"Authorization": "Bearer super-secret-key"},
    )
    assert resp.status_code == 200

    # The exchange was recorded, and the credential was NOT persisted.
    exchanges = store.read_all()
    assert len(exchanges) == 1
    ex = exchanges[0]
    assert ex.api == "openai"
    assert ex.turn == 1
    assert ex.response.assembled["content"] == "The capital is Paris."
    assert "authorization" not in ex.request.headers_subset

    # Emit + lint
    text, warnings = emit_yaml(exchanges, "balanced")
    assert lint_text(text) == []


def test_e2e_anthropic_streaming_and_error(patched_proxy):
    client, store, tmp_path = patched_proxy
    stream_body = (
        'event: content_block_start\n'
        'data: {"type":"content_block_start","index":0,'
        '"content_block":{"type":"text","text":""}}\n\n'
        'event: content_block_delta\n'
        'data: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"text_delta","text":"streamed hi"}}\n\n'
        'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )
    _StubClient._queue = [
        _StubResponse(200, stream_body, {"content-type": "text/event-stream"}),
        _StubResponse(429, json.dumps({"error": {"message": "rate limited"}}),
                      {"content-type": "application/json"}),
    ]
    # turn 1: streaming success
    r1 = client.post(
        "/v1/messages",
        json={"model": "claude-3-5-sonnet", "stream": True,
              "messages": [{"role": "user", "content": "say hi via stream please"}]},
    )
    assert r1.status_code == 200
    # turn 2: same conversation, an injected 429
    r2 = client.post(
        "/v1/messages",
        json={"model": "claude-3-5-sonnet",
              "messages": [{"role": "user", "content": "say hi via stream please"},
                           {"role": "assistant", "content": "streamed hi"},
                           {"role": "user", "content": "again"}]},
    )
    assert r2.status_code == 429

    exchanges = store.read_all()
    assert len(exchanges) == 2
    assert exchanges[0].api == "anthropic"
    assert exchanges[0].response.assembled["content"] == "streamed hi"
    # both turns share one conversation (keyed on first user message)
    assert exchanges[0].conversation_id == exchanges[1].conversation_id
    assert exchanges[0].turn == 1 and exchanges[1].turn == 2
    # the error turn recorded a non-200 status
    assert exchanges[1].response.status == 429

    text, _ = emit_yaml(exchanges, "balanced")
    assert "status: 429" in text
    assert lint_text(text) == []
