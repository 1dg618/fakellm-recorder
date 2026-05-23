"""The session store: normalized exchange records as append-only JSONL.

One file per recording session. Greppable, diffable, trivially appendable while
the proxy runs.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


@dataclass
class RequestRecord:
    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = field(default_factory=list)
    stream: bool = False
    headers_subset: dict[str, str] = field(default_factory=dict)


@dataclass
class ResponseRecord:
    status: int
    assembled: dict[str, Any] = field(default_factory=dict)  # {content?, tool_calls?}
    raw_stream: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


@dataclass
class Exchange:
    conversation_id: str
    turn: int
    api: str  # "openai" | "anthropic"
    request: RequestRecord
    response: ResponseRecord
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Exchange":
        return cls(
            conversation_id=d["conversation_id"],
            turn=d["turn"],
            api=d["api"],
            request=RequestRecord(**d["request"]),
            response=ResponseRecord(**d["response"]),
            ts=d.get("ts", ""),
        )


class SessionStore:
    """Append-only JSONL writer/reader for one recording session."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, exchange: Exchange) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(exchange.to_json() + "\n")

    def read_all(self) -> list[Exchange]:
        return list(self.iter_exchanges())

    def iter_exchanges(self) -> Iterator[Exchange]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield Exchange.from_dict(json.loads(line))
