# A5 - Global Pentagonal Geospatial Index (Python Port)

## Overview
A5 is a Python library that partitions the world into equal-area pentagonal cells at 31 resolution levels. Built on a dodecahedral geometry, it provides millimeter-accurate geospatial indexing (30mm² at highest resolution) encoded as 64-bit integers.

Website: https://a5geo.org
Docs: ../a5/docs/api-reference/README.md

## Ports

- A5 is ported to Python and Rust, with each language being treated as an equally valid port.
- The TypeScript project contains the docs and website
- The projects are typically checked out in parallel and can be accessed via:
  - `../a5` TypeScript version
  - `../a5-py` Python version
  - `../a5-rs` Rust version
- Each of the ports has its own `CLAUDE.md` file. Whenever you work with another project, read this file to get the additional context.


## Python Project Structure
- `/a5` - Python source code organized into modules:
  - `/core` - Core geospatial functionality (cell, hex, hilbert, serialization, etc.)
  - `/math` - Mathematical primitives (vec2, vec3, quat)
  - `/geometry` - Geometric calculations (pentagon, spherical_triangle, spherical_polygon)
  - `/projections` - Map projection implementations (dodecahedron, authalic, gnomonic, etc.)
- `/tests` - pytest test files organized by module
  - `/core/fixtures` - Test data fixtures for core module
  - `/geometry/fixtures` - Test data fixtures for geometry module
  - `/projections/fixtures` - Test data fixtures for projections module
  - `/fixtures` - Shared test fixtures
- `/examples/wireframe` - CLI applications demonstrating A5 usage
- `/dist` - Built distributions (wheels and source distributions)

## Key Concepts
- **Cell**: A pentagonal region at a specific resolution (represented as int)
- **Resolution**: 0-30, where 0 is global coverage and 30 is ~30mm²
- **Compaction**: Combining child cells into parent cells for efficient storage
- **Cell ID**: Always an int (use `u64_to_hex()` for string representation)

## Commands
```bash
# Setup (requires uv: https://docs.astral.sh/uv/)
uv pip install -e ".[test]"   # Install with test dependencies

# Testing
uv run pytest                 # Run all tests
uv run pytest tests/core/     # Run tests in specific directory
uv run pytest tests/test_cell.py  # Run specific test file
uv run pytest -k "test_name"  # Run tests matching pattern
uv run pytest -v              # Verbose output

# Building & Publishing
rm -rf dist/*
uv version --bump patch|minor|major
uv build
uv publish
```

## Development Guidelines
- **Python**: Source files in `/a5`, requires Python >=3.8
- **Tests**: Use pytest, organized by module with fixture-driven tests
- **Package Manager**: Uses `uv` for dependency management and builds
- **Package**: Package name is `pya5`, import as `import a5`
- **Cell IDs**: Always use int internally, convert to hex with `u64_to_hex()` / `hex_to_bigint()`
- **Type Hints**: Use type hints throughout for better code clarity
- **Dependencies**: Core library has no runtime dependencies
- **Coordinate Types**: LonLat is just a tuple `(float, float)` in Python - no special casting needed

## Porting from TypeScript
When porting features from TypeScript:
1. **Reference implementation**: TypeScript in `../a5` is the source of truth
2. **Key file mappings**:
   - `modules/core/cell.ts` → `a5/core/cell.py`
   - `tests/cell.test.ts` → `tests/core/test_cell.py`
   - `modules/index.ts` → `a5/__init__.py`
3. **Type conversions**:
   - `bigint` → `int`
   - `LonLat` branded type → `tuple[float, float]`
   - `Result<T, Error>` → direct return or raise exception
4. **Test patterns**: Follow existing test structure with `class Test*` grouping tests

## CI Checks (run as a final verification)
```bash
# 1. Install dependencies
uv pip install -e ".[test]"

# 2. Run tests
uv run pytest
```

These are the same checks that run in CI (.github/workflows/test.yml). Run these to verify your changes before the user reviews the code.

## Git Usage

- **DO** use git commands for debugging and information gathering:
  - `git status` - Check current state
  - `git diff` - Compare changes
  - `git log` - View commit history
  - `git diff master` - Compare to master branch
  - `git show <commit>` - View specific commits
- **DO NOT** create git commits - the user will review the code and commit it themselves

## Testing Strategy

- Tests are written such that they can easily be ported to other languages
- Tests should be driven by fixtures, JSON files that specify known input & output values
- When porting tests from TypeScript:
  - 1. Copy the fixture files from the TypeScript repo
  - 2. Create test files that load and verify against the fixtures
  - 3. Run tests and verify exact matches
- Python port should NOT create its own fixture generators, copy them from TypeScript
- IMPORTANT: The ports should verify that the behavior is exactly the same, it is NOT acceptable to round values or accept approximate equality

## Important
- Keep changes minimal and focused on requested tasks
- Verify imports are clean and no circular dependencies exist
- Follow PEP 8 style guidelines
- Test fixtures should be copied from TypeScript repo, not regenerated
- Ensure exact numerical equivalence with TypeScript implementation
- If instructions in `CLAUDE.md` seem wrong, update them and notify the user
