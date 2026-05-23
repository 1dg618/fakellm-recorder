"""fakellm-recorder CLI: proxy / emit / lint."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from . import __version__
from .emitter import emit_yaml
from .linter import lint_text
from .proxy import build_app
from .scrub import Scrubber
from .store import SessionStore


def _cmd_proxy(args: argparse.Namespace) -> int:
    import uvicorn

    store = SessionStore(args.out)
    scrubber = Scrubber(enabled=not args.no_scrub)
    app = build_app(store, scrubber, args.upstream)
    print(f"fakellm-recorder proxy → upstream={args.upstream}")
    print(f"recording to {args.out}")
    print(f"point your SDK base_url at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def _cmd_emit(args: argparse.Namespace) -> int:
    store = SessionStore(args.session)
    exchanges = store.read_all()
    if not exchanges:
        print(f"no exchanges found in {args.session}", file=sys.stderr)
        return 1
    text, warnings = emit_yaml(exchanges, strictness=args.match_strictness)
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"wrote {args.out} ({len(exchanges)} exchanges)")
    if warnings:
        print("\nwarnings:")
        for w in warnings:
            print(f"  - {w}")
    return 0


def _cmd_lint(args: argparse.Namespace) -> int:
    text = Path(args.config).read_text(encoding="utf-8")
    issues = lint_text(text)
    if not issues:
        print(f"{args.config}: clean")
        return 0
    print(f"{args.config}: {len(issues)} issue(s)")
    for issue in issues:
        print(f"  - {issue}")
    return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fakellm-recorder")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("proxy", help="start recording proxy")
    pp.add_argument("--upstream", choices=["openai", "anthropic", "auto"], default="auto")
    pp.add_argument("--host", default="127.0.0.1")
    pp.add_argument("--port", type=int, default=8888)
    pp.add_argument("--out", default="sessions/run.jsonl")
    pp.add_argument("--no-scrub", action="store_true", help="disable PII scrubbing (creds still stripped)")
    pp.set_defaults(func=_cmd_proxy)

    pe = sub.add_parser("emit", help="emit fakellm.yaml from a session file")
    pe.add_argument("session")
    pe.add_argument("--out", default="fakellm.yaml")
    pe.add_argument(
        "--match-strictness", choices=["loose", "balanced", "strict"], default="balanced"
    )
    pe.set_defaults(func=_cmd_emit)

    pl = sub.add_parser("lint", help="lint a fakellm.yaml")
    pl.add_argument("config")
    pl.set_defaults(func=_cmd_lint)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
