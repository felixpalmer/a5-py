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

## Publish (for maintainers)

### Git strategy

Prereleases run from `main`, stable from the `*-release` branches.
Each minor version gets a branch, e.g. `1.2-release` which is cut from `main`:

```bash
git checkout main
git pull
git checkout -b 1.2-release
```

PRs are merged to `main` and then cherry-picked to the latest release branch (in principle to older releases also, but this is rare).

```bash
git checkout 1.2-release
git cherry-pick 1234abcd
```

### Publishing to PyPI

`./publish.sh` tags `v<version>` and pushes; CI builds, tests, and publishes pya5 to PyPI via
trusted publishing (OIDC) — no API token and no local `uv publish`.

```bash
# Bump the version, e.g. uv version --bump patch  (or edit pyproject.toml: 1.0.0b1)
uv lock   # refreshes uv.lock with the new version — required, else the lockfile is stale
# Add a "#### pya5 [v<version>] - <date>" entry to CHANGELOG.md
git add pyproject.toml uv.lock CHANGELOG.md
git commit -m "x.y.z release"

./publish.sh beta   # prerelease (PEP 440, e.g. 1.0.0b1), from main
./publish.sh prod   # stable X.Y.Z, from a *-release branch
```
