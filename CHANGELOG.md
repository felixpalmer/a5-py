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

## pya5 v0.5

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
