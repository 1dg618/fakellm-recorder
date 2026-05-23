"""Parse and assemble streamed responses from both API dialects.

We keep the raw event list (so a later streaming-fidelity mode can replay chunk
by chunk) AND assemble the final content / tool_calls (so the default emitted
rule is simple).

OpenAI: `data: {...}` lines terminated by `data: [DONE]`.
Anthropic: typed events — message_start, content_block_start,
content_block_delta, content_block_stop, message_delta, message_stop.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AssembledResponse:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw_events: list[dict[str, Any]] = field(default_factory=list)

    def to_respond(self) -> dict[str, Any]:
        """Shape into a fakellm `respond:` block (content/tool_calls only)."""
        out: dict[str, Any] = {}
        if self.content:
            out["content"] = self.content
        if self.tool_calls:
            out["tool_calls"] = self.tool_calls
        return out


def parse_sse_lines(raw: str) -> list[tuple[str | None, str]]:
    """Parse raw SSE text into (event_type, data) pairs.

    OpenAI omits an explicit event: line, so event_type is None there.
    """
    events: list[tuple[str | None, str]] = []
    event_type: str | None = None
    data_buf: list[str] = []
    for line in raw.splitlines():
        if line.startswith("event:"):
            event_type = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_buf.append(line[len("data:"):].strip())
        elif line == "":
            if data_buf:
                events.append((event_type, "\n".join(data_buf)))
            event_type = None
            data_buf = []
    if data_buf:
        events.append((event_type, "\n".join(data_buf)))
    return events


def assemble_openai(raw: str) -> AssembledResponse:
    result = AssembledResponse()
    # tool call index -> {"name": str, "arguments": str}
    tool_acc: dict[int, dict[str, str]] = {}
    for _evt, data in parse_sse_lines(raw):
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        result.raw_events.append(chunk)
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            if isinstance(delta.get("content"), str):
                result.content += delta["content"]
            for tc in delta.get("tool_calls", []) or []:
                idx = tc.get("index", 0)
                slot = tool_acc.setdefault(idx, {"name": "", "arguments": ""})
                fn = tc.get("function", {})
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
    for idx in sorted(tool_acc):
        slot = tool_acc[idx]
        result.tool_calls.append(_finalize_tool_call(slot["name"], slot["arguments"]))
    return result


def assemble_anthropic(raw: str) -> AssembledResponse:
    result = AssembledResponse()
    # content block index -> {"type", "name", "input_json"/"text"}
    blocks: dict[int, dict[str, Any]] = {}
    for evt, data in parse_sse_lines(raw):
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        result.raw_events.append(payload)
        etype = evt or payload.get("type")
        if etype == "content_block_start":
            idx = payload.get("index", 0)
            cb = payload.get("content_block", {})
            blocks[idx] = {
                "type": cb.get("type"),
                "name": cb.get("name", ""),
                "input_json": "",
                "text": "",
            }
        elif etype == "content_block_delta":
            idx = payload.get("index", 0)
            slot = blocks.setdefault(
                idx, {"type": None, "name": "", "input_json": "", "text": ""}
            )
            delta = payload.get("delta", {})
            if delta.get("type") == "text_delta":
                slot["text"] += delta.get("text", "")
            elif delta.get("type") == "input_json_delta":
                slot["input_json"] += delta.get("partial_json", "")
    for idx in sorted(blocks):
        slot = blocks[idx]
        if slot["type"] == "text":
            result.content += slot["text"]
        elif slot["type"] == "tool_use":
            result.tool_calls.append(
                _finalize_tool_call(slot["name"], slot["input_json"])
            )
    return result


def _finalize_tool_call(name: str, arguments_json: str) -> dict[str, Any]:
    """Normalize a tool call into fakellm's {name, arguments} shape.

    fakellm wants arguments as a mapping; both APIs stream them as a JSON string.
    """
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError:
        args = {"_raw": arguments_json}
    return {"name": name, "arguments": args}


def assemble_nonstreaming(api: str, body: dict[str, Any]) -> AssembledResponse:
    """Assemble a normal (non-streamed) JSON response body."""
    result = AssembledResponse()
    if api == "openai":
        for choice in body.get("choices", []):
            msg = choice.get("message", {})
            if isinstance(msg.get("content"), str):
                result.content += msg["content"]
            for tc in msg.get("tool_calls", []) or []:
                fn = tc.get("function", {})
                result.tool_calls.append(
                    _finalize_tool_call(fn.get("name", ""), fn.get("arguments", ""))
                )
    elif api == "anthropic":
        for block in body.get("content", []):
            if block.get("type") == "text":
                result.content += block.get("text", "")
            elif block.get("type") == "tool_use":
                result.tool_calls.append(
                    {"name": block.get("name", ""), "arguments": block.get("input", {})}
                )
    return result
