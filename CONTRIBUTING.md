# Contributing

Thanks for your interest in fakellm-recorder. Issues and pull requests are welcome.

## Development setup

```bash
git clone https://github.com/1dg618/fakellm-recorder.git
cd fakellm-recorder
pip install -e ".[dev]"
```

## Running tests

```bash
python -m pytest
```

Please make sure the test suite passes before opening a pull request, and add
tests for any new behavior.

## Reporting bugs

Open an issue with a clear description, the steps to reproduce, and what you
expected to happen instead.

## Pull requests

- Keep changes focused; one logical change per PR.
- Match the existing code style.
- Update `CHANGELOG.md` under the `[Unreleased]` section.
