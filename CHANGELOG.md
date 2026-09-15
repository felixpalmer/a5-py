# Change Log

All notable changes to pya5 will be documented in this file.

For the latest documentation, visit [A5 Documentation](https://a5geo.org)

<!--
Each version should:
  List its release date in the above format.
  Group changes to describe their impact on the project, as follows:
  Added for new features.
  Changed for changes in existing functionality.
  Deprecated for once-stable features removed in upcoming releases.
  Removed for deprecated features removed in this release.
  Fixed for any bug fixes.
  Security to invite users to upgrade in case of vulnerabilities.
Ref: http://keepachangelog.com/en/0.3.0/
-->

## pya5

#### pya5 [v0.11.0] - unreleased

##### Added

- Optional Rust backend via PyO3 bindings to the [a5-rs](https://github.com/felixpalmer/a5-rs)
  crate, covering the whole public API. Roughly two orders of magnitude faster than
  pure Python. Opt in with `A5_BACKEND=rust`; `a5.get_backend()` reports which
  implementation is live.
- `a5.batch` with sequence-taking variants of `cell_to_parent`, `cell_to_children`,
  `get_resolution` and `cell_area`, which amortise the Python/Rust boundary
  crossing over a whole batch. Available on both backends.
- Platform wheels (manylinux, musllinux, macOS x86_64/arm64, Windows) bundling
  the compiled backend, built as `abi3` so one wheel per platform serves
  CPython 3.9 and later.
- Differential test suite comparing the two backends cell-for-cell, plus a CI
  matrix that runs the entire fixture suite once per backend.

##### Changed

- The pure-Python implementation remains the default in 0.x and stays the
  reference implementation. The default becomes `auto` (compiled where
  available) in 1.0.
- Minimum Python is now 3.9, up from 3.8, which reached end of life in October
  2024. This is the floor for the `abi3` wheels.
- Building from source now requires a Rust toolchain, as the build backend is
  maturin rather than hatchling. Installing a platform wheel does not.

#### pya5 [v0.10.0] - August 28 2026

- Improve precision in vec3.angle function (#59)
- Feat: polygonToCells overlapping containment (#58)
- feat: Add cellEdgeLengthAvg function (#57)
- feat: More efficient equal area projection (#54)
- feat: Use L-system to layout lattice curve (#53)

#### pya5 [v0.9.0] - June 17 2026

- **BREAKING**: Remove cell_to_spherical
- chore: align API (#50)
- Support holes (#49)
- Feat: refactor to use EqualAreaProjection (#48)
- feat: Faster polyhedral projection (#47)

#### pya5 [v0.8.0] - May 12 2026

- fix(cell): Spiral unstable in polar regions (#46)
- feat(regions): Implement polygonToCells (#45)
- feat(traversal): lineStringToCells (#44)
- Refactor: lattice boundary helpers (#43)
- feat(serialization): Improve cellToParent speed (#42)

#### pya5 [v0.7.3] - Apr 9 2026

- fix: normalize longitude in toLonLat to -180-180 (#40)

#### pya5 [v0.7.2] - Mar 29 2026

- Feature: Support neighbor functions in res 0 & 1 (#38)

#### pya5 [v0.7.1] - Mar 10 2026

- Feature: Support (de)serialization of resolution 30 cells (#37)

#### pya5 [v0.7.0] - Mar 3 2026

- Feature: gridDisk & sphericalCap (#35)

#### pya5 [v0.6.1] - Nov 11 2025

- feat: World cell handling (#34)

#### pya5 [v0.6.0] - Oct 30 2025

- Feature: cell compaction/uncompaction (#33)

#### pya5 [v0.5.0] - Sep 21 2025

- **BREAKING**: Renamed hex conversion functions to use u64 naming convention (#30)
  - `hex_to_bigint` → `hex_to_u64`
  - `bigint_to_hex` → `u64_to_hex`

## pya5 v0.4

#### pya5 [v0.4.2] - Aug 7 2025

- Fixed: cell_to_children and cell_to_parent functions (#29)

#### pya5 [v0.4.1] - Jul 30 2025

- Removed: numpy dependency (#27)
- Added: Port cell functions and wireframe test script (#26)

#### pya5 [v0.4.0] - Jul 15 2025

- Added: Dodecahedron projection port (#25)
- Added: Port CRS and polyhedral projection (#24)
- Changed: Re-port serialization (#23)
- Changed: Move PentagonShape class (#22)
- Changed: Update hilbert curve to include _shift_digits (#21)
- Changed: Convert authalic functions to AuthalicProjection class (#20)
- Added: Support for Barycentric coordinates (#19)
- Changed: Port gnomonic to new class based GnomonicProjection (#18)
- Changed: Re-port spherical-polygon & add vector.py (#17)
- Changed: Re-port constants & authalic (#16)

## pya5 v0.3

#### pya5 [v0.3.0] - May 27 2025

- Added: Porting serialization (#14)
- Added: utils.py, test_utils.py (#12)
- Added: Porting dodecahedron (#13)
- Added: Porting triangle (#10)
- Added: License
- Changed: Tidy quat_from_spherical
- Changed: Move quat helpers
- Added: rotationTo function
- Fixed: Quaternions implementation
- Changed: Rename, origin.py and coordinate_systems.py
- Added: Porting origin function
- Changed: Update hilbert.py
- Added: Hilbert curve implementation

## pya5 v0.2

#### pya5 [v0.2.0] - May 18 2025

- Added: CI testing workflow (#4)
- Added: Publishing configuration (#2)
- Added: PR template (#3)
- Added: Warp functionality
- Added: Gnomonic conversion and tests
- Added: Math functions port
- Changed: Update contributing documentation

## pya5 v0.1

#### pya5 [v0.1.0] - May 16 2025

- Added: Initial Python port of A5 - Global Pentagonal Geospatial Index
- Added: Simple hex.py implementation
- Added: Project initialization
