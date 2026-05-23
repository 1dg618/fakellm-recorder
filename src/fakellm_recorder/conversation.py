"""Conversation identity, mirroring fakellm's own bucketing.

fakellm keys a conversation on a stable hash of the first user message, and lets
a client override it via the X-Fakellm-Conversation-Id header. We replicate that
exactly so the turn numbers we record line up with what fakellm sees at replay.
"""
from __future__ import annotations

import hashlib
from typing import Any

CONVERSATION_HEADER = "x-fakellm-conversation-id"


def first_user_text(messages: list[dict[str, Any]]) -> str:
    """Extract the text of the first user message.

    Both APIs allow content to be a plain string or a list of content blocks.
    We normalize to a single string so the hash is stable across shapes.
    """
    for msg in messages:
        if msg.get("role") == "user":
            return _content_to_text(msg.get("content"))
    return ""


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                # OpenAI: {"type": "text", "text": "..."}
                # Anthropic: same shape for text blocks
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block.get("text"), str):
                    parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def conversation_id(messages: list[dict[str, Any]], header_value: str | None) -> str:
    """Resolve the conversation id for a request.

    An explicit header always wins (tests want deterministic control). Otherwise
    we hash the first user message, matching fakellm's default scheme.
    """
    if header_value:
        return header_value
    seed = first_user_text(messages)
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return digest[:12]
