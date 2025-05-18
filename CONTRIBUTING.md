# Contributing to A5-py

Thank you for contributing to the Python version of [A5](https://a5geo.org). We are actively looking for new contributors.

## Setting up environment

First install [uv](https://docs.astral.sh/uv/)

```bash
# Install test dependencies
uv pip install -e ".[test]"
```

## Run tests

```bash
uv run pytest
```

## Build & publish

```bash
uv build
uv publish
```