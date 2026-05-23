"""The recording proxy.

Byte-faithful forwarding to the real upstream in record mode; all cleverness
lives in the emitter, not here. Captures request + response (including assembled
streaming) and tees a normalized Exchange to the session store.
"""
from __future__ import annotations

import json
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .conversation import CONVERSATION_HEADER, conversation_id
from .scrub import Scrubber
from .store import Exchange, RequestRecord, ResponseRecord, SessionStore
from .streaming import (
    assemble_anthropic,
    assemble_nonstreaming,
    assemble_openai,
)

UPSTREAMS = {
    "openai": "https://api.openai.com",
    "anthropic": "https://api.anthropic.com",
}

# Headers we must NOT forward verbatim (host gets recomputed by httpx).
_HOP_BY_HOP = {"host", "content-length", "connection", "accept-encoding"}


class ProxyState:
    """Holds turn counters per conversation for this recording session."""

    def __init__(self, store: SessionStore, scrubber: Scrubber, upstream_mode: str):
        self.store = store
        self.scrubber = scrubber
        self.upstream_mode = upstream_mode  # "openai" | "anthropic" | "auto"
        self._turns: dict[str, int] = {}

    def next_turn(self, conv_id: str) -> int:
        self._turns[conv_id] = self._turns.get(conv_id, 0) + 1
        return self._turns[conv_id]


def _api_for_path(path: str, mode: str) -> str:
    if mode in ("openai", "anthropic"):
        return mode
    # auto: route by path
    if path.endswith("/messages"):
        return "anthropic"
    return "openai"


def _forward_headers(headers: dict[str, str], api: str) -> dict[str, str]:
    out = {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}
    return out


async def _handle(request: Request, state: ProxyState) -> Response:
    body_bytes = await request.body()
    try:
        payload: dict[str, Any] = json.loads(body_bytes) if body_bytes else {}
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    api = _api_for_path(request.url.path, state.upstream_mode)
    upstream_base = UPSTREAMS[api]
    messages = payload.get("messages", [])
    incoming_headers = {k: v for k, v in request.headers.items()}
    conv_id = conversation_id(messages, incoming_headers.get(CONVERSATION_HEADER))
    turn = state.next_turn(conv_id)
    is_stream = bool(payload.get("stream"))

    url = upstream_base + request.url.path
    fwd_headers = _forward_headers(incoming_headers, api)

    async with httpx.AsyncClient(timeout=120.0) as client:
        upstream_resp = await client.request(
            request.method,
            url,
            content=body_bytes,
            headers=fwd_headers,
            params=dict(request.query_params),
        )

    raw_text = upstream_resp.text
    status = upstream_resp.status_code

    # Assemble for the record.
    if status >= 400:
        assembled = {}
        error_text = raw_text
        raw_stream: list[dict[str, Any]] = []
    elif is_stream:
        ar = assemble_openai(raw_text) if api == "openai" else assemble_anthropic(raw_text)
        assembled = ar.to_respond()
        raw_stream = ar.raw_events
        error_text = None
    else:
        try:
            resp_body = json.loads(raw_text)
        except json.JSONDecodeError:
            resp_body = {}
        ar = assemble_nonstreaming(api, resp_body)
        assembled = ar.to_respond()
        raw_stream = []
        error_text = None

    # Scrub before persisting.
    sc = state.scrubber
    exchange = Exchange(
        conversation_id=conv_id,
        turn=turn,
        api=api,
        request=RequestRecord(
            model=payload.get("model", ""),
            messages=sc.scrub_obj(messages),
            tools=payload.get("tools", []) or [],
            stream=is_stream,
            headers_subset=sc.filter_headers(incoming_headers),
        ),
        response=ResponseRecord(
            status=status,
            assembled=sc.scrub_obj(assembled),
            raw_stream=sc.scrub_obj(raw_stream),
            error=sc.scrub_text(error_text) if error_text else None,
        ),
    )
    state.store.append(exchange)

    # Return the upstream response to the client byte-faithfully.
    passthrough_headers = {
        k: v
        for k, v in upstream_resp.headers.items()
        if k.lower() not in {"content-encoding", "content-length", "transfer-encoding"}
    }
    return Response(
        content=upstream_resp.content,
        status_code=status,
        headers=passthrough_headers,
        media_type=upstream_resp.headers.get("content-type"),
    )


def build_app(store: SessionStore, scrubber: Scrubber, upstream_mode: str) -> Starlette:
    state = ProxyState(store, scrubber, upstream_mode)

    async def handler(request: Request) -> Response:
        return await _handle(request, state)

    routes = [
        Route("/v1/chat/completions", handler, methods=["POST"]),
        Route("/v1/messages", handler, methods=["POST"]),
    ]
    return Starlette(routes=routes)
