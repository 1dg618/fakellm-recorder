# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-23
### Added
- Recording proxy that forwards to the real OpenAI/Anthropic APIs and tees captured traffic to a session store.
- Rule emitter that turns recorded sessions into `fakellm.yaml`, with inverse-frequency n-gram matching and `loose` / `balanced` / `strict` strictness modes.
- Multi-turn conversation bucketing that mirrors fakellm so turn numbers line up at replay.
- Streaming assembly for both the OpenAI and Anthropic SSE dialects.
- Credential stripping at capture and PII scrubbing on by default.
- Linter for unreachable rules and config typos.
- CLI with `proxy`, `emit`, and `lint` subcommands.

[Unreleased]: https://github.com/1dg618/fakellm-recorder/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/1dg618/fakellm-recorder/releases/tag/v0.1.0
