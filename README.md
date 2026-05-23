# fakellm-recorder

Record real OpenAI/Anthropic API traffic and emit [`fakellm`](https://pypi.org/project/fakellm/) `fakellm.yaml` rules automatically.

VCR-style zero-effort capture, combined with fakellm's editable, error-injectable
YAML rules. Run your real code against the real API once through the proxy, get a
`fakellm.yaml` out, hand-edit it to add the error paths recordings can't capture,
and commit it.

## Install

```bash
pip install fakellm-recorder    # once published
# or, from source:
pip install -e .
```

## The loop

**1. Record.** Start the proxy and point your SDK's `base_url` at it, then run
your existing test or script once.

```bash
fakellm-recorder proxy --upstream auto --out sessions/run1.jsonl
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8888/v1", api_key="sk-...real-key...")
# run_my_agent(client, ...)  # real traffic flows through and is recorded
```

The proxy forwards byte-faithfully to the real upstream, so your code behaves
exactly as in production. Credentials are stripped before anything is written to
disk; prompts/completions are scrubbed for emails and obvious keys by default.

**2. Emit.** Turn the recorded session into rules.

```bash
fakellm-recorder emit sessions/run1.jsonl --out fakellm.yaml --match-strictness balanced
```

**3. Edit + commit.** Add the 429s/500s/malformed responses recordings can't
capture, then commit `fakellm.yaml` alongside your tests. At replay time it's
plain fakellm — no recorder needed.

```bash
fakellm serve   # from the fakellm package
```

## Match strictness

The hard part is choosing a `messages_contain` substring specific enough to fire
on the right turn but loose enough not to only match that exact transcript. The
emitter ranks candidate n-grams by inverse frequency across the whole session, so
shared boilerplate (e.g. "You are a helpful assistant") is never chosen.

| Mode | What it emits |
| --- | --- |
| `loose` | mostly `turn:` + a model glob (`gpt-4*`); "any response will do" tests |
| `balanced` (default) | `turn:` + one distinctive substring (or a `tool_result_contains` anchor on post-tool turns) |
| `strict` | turn + substring + exact `model_matches` + `tools_include`; closest to a faithful replay |

## Lint

A standalone check for unreachable rules (shadowed by an earlier first-match),
unknown condition/response keys, and responses that set neither `content` nor
`tool_calls` unintentionally.

```bash
fakellm-recorder lint fakellm.yaml
```

## Security

- **Credentials are never persisted.** Header capture is allowlist-based, so
  `Authorization` / `x-api-key` / auth headers are dropped at capture time even
  if PII scrubbing is disabled.
- **PII scrubbing is on by default** (emails, OpenAI/Anthropic keys, bearer
  tokens) because the generated YAML is a commitable artifact. Disable with
  `--no-scrub` (credentials are still stripped). Add your own regexes in code via
  `Scrubber(custom_patterns=[...])`.

## Caveats

- **fakellm is a new, single-author beta** (0.3.x). Emitted files are stamped
  with the targeted config version; pin a fakellm version and expect schema churn.
- **Streaming reconstruction** handles both SSE dialects and keeps raw events for
  a future chunk-fidelity replay mode, but assembly of exotic event orderings may
  need tuning — the raw events are retained so you can fix assembly without
  re-recording.
- **Single worker.** fakellm stores state in process memory; keep both it and
  this proxy single-worker.

## License

MIT
