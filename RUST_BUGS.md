# Cross-port divergences found by the differential suite

`tests/test_differential.py` runs the pure-Python and compiled backends over the
same generated corpus and compares them. Everything below is a case where they
disagree. Each needs fixing in its home repository rather than being papered
over in `a5/_native.py`.

Observed against a5-rs `7ad0978` (0.10.0), the revision `Cargo.toml` names.

| # | what | where the fix belongs | status | pinned by a test |
| --- | --- | --- | --- | --- |
| 1 | `get_num_cells` wrong at resolutions 28-30 | a5-rs | [PR #65](https://github.com/felixpalmer/a5-rs/pull/65) open | yes, strict xfail |
| 2 | `hex_to_u64` accepts too much | a5-py (+ a decision for a5-js) | open | yes |
| 3 | `cell_to_parent` unchecked above the cell's resolution | a5-rs, a5-py, a5-js | open | partly |
| 4 | pole results depend on call history | all ports | open, not characterised | no |

---

## 1. a5-rs: `get_num_cells` returns wrong values for resolutions 28-30

**Status: fix proposed — [a5-rs#65](https://github.com/felixpalmer/a5-rs/pull/65),
open, awaiting review.** Head `12da7f5`, all 15 checks green. Not merged, so the
`rev` in `Cargo.toml` still points at `7ad0978` and the divergence is still live
here. See "Landing the fix" at the end of this entry.

**Severity:** wrong results from a public API function.

`src/core/cell_info.rs` special-cases the three highest resolutions:

```rust
// Match JavaScript's precision behavior exactly
// For resolution 28, JavaScript returns 1080863910568919000 due to precision loss
if resolution == 28 { return 1080863910568919000; }
if resolution == 29 { return 4323455642275676000; }
if resolution == 30 { return 17293822569102705000; }
```

| resolution | a5-rs returns | correct (`60 * 4^(r-1)`) | off by |
| --- | ---: | ---: | ---: |
| 28 | 1080863910568919000 | 1080863910568919040 | 40 |
| 29 | 4323455642275676000 | 4323455642275676160 | 160 |
| 30 | 17293822569102705000 | 17293822569102704640 | 360 |

**The premise of the comment is false.** There is no precision loss to mirror.
All three correct values are *exactly* representable as an IEEE-754 double —
they are `15 * 2^(2r)`, which needs four significant bits — so JavaScript's
arithmetic is exact here. What the comment describes as "what JavaScript
returns" is JavaScript's *shortest round-trip decimal printing* of that exact
double. `1080863910568919000` and `1080863910568919040` denote the same double;
they are different integers.

So the literals were transcribed from JS console output and frozen as `u64`,
which is the one type that can represent the value exactly.

**Corroboration:** the shared fixture `tests/fixtures/cell-info.json` carries
both renderings — `count` (the JS number, printed short) and `countBigInt` (the
exact value). The TypeScript source has two overloads, `getNumCells(number):
number` and `getNumCells(bigint): bigint`. A `u64` return is the bigint
contract, so a5-rs should implement the bigint branch. a5-py does, and asserts
against `countBigInt`; a5-rs's own `tests/cell_info.rs` reads `count`, which is
why its suite is green.

**Fix:** delete the three special cases; the general
`60 * 4_u64.pow((resolution - 1) as u32)` already returns the right answer for
28-30 with no overflow (max is ~1.7e19, `u64::MAX` is ~1.8e19). Then switch
`tests/cell_info.rs` to deserialise `countBigInt`.

This is what a5-rs#65 does: it drops the three `if resolution == …` branches and
reads the fixture's `countBigInt` string, parsing it to `u64`. No fixture
regeneration is needed — `cell-info.json` has always carried both fields.

**Knock-on effects:**

- `get_num_children(parent, child)` inherits it whenever
  `parent < FIRST_HILBERT_RESOLUTION` and `child >= 28`, since that branch
  divides two `get_num_cells` results. The aperture-4 shortcut used above the
  first Hilbert resolution is unaffected.
- `cell_area` is **not** affected: it divides `AUTHALIC_AREA_EARTH` by the
  count, and both integers convert to the same `f64`, so the quotient is
  identical. Verified across all resolutions.

**Pinned by:** `tests/test_cell_info.py::test_get_num_cells_returns_correct_count_for_all_resolutions`
(strict xfail on the rust backend) and `tests/test_differential.py::test_known_divergences`.

**Landing the fix.** Both pins assert the bug is *present*, so bumping the
dependency without removing them turns the suite red — deliberately, so the
cleanup cannot be forgotten. Once a5-rs#65 merges, in one commit:

1. `Cargo.toml` — set `rev` to the merge commit, then `cargo update -p a5`.
2. `tests/test_cell_info.py` — drop the `@pytest.mark.xfail` and the now-unused
   `get_backend` import.
3. `tests/test_differential.py` — delete `test_known_divergences`, widen
   `test_get_num_cells` to `range(-1, 31)`, and remove the
   `parent < FIRST_HILBERT_RESOLUTION and child >= 28` skip in
   `test_get_num_children`.
4. Delete this entry.

Then run the suite on both backends; `test_get_num_cells` and
`test_get_num_children` should pass across the full resolution range with no
exclusions.

---

## 2. a5-py: `hex_to_u64` accepts input the other two ports reject

**Severity:** low, but it is a5-py that is out of line.

`a5/core/hex.py` is `int(hex_str, 16)`, which is far more permissive than the
reference:

| input | TypeScript (BigInt of `0x` + hex) | a5-rs `u64::from_str_radix` | a5-py `int(s, 16)` |
| --- | --- | --- | --- |
| `"1f"` | 31 | 31 | 31 |
| `"0x1f"` | SyntaxError | Err | **31** |
| `"1_f"` | SyntaxError | Err | **31** |
| `" 1f "` | SyntaxError | Err | **31** |
| `"-1"` | SyntaxError | Err | **-1** |
| `"10000000000000000"` | 18446744073709551616 | Err | 18446744073709551616 |

a5-rs matches TypeScript on the first five rows. Only the last differs, and
rejecting a value that cannot fit a `u64` is defensible for a `u64` API.

**Fix (a5-py):** validate before converting, so all three ports agree. This is a
behaviour change for anyone currently passing `0x`-prefixed strings, so it wants
its own PR and a changelog note rather than riding along here.

**Pinned by:** `tests/test_differential.py::TestIndexing::test_hex_leniency_divergence`.

---

## 3. Both ports: `cell_to_parent` is unchecked above the cell's own resolution

**Severity:** low; out-of-domain input, no fixture covers it.

Asking for a parent at a resolution *finer* than the cell's own is meaningless.
Neither port rejects it:

- a5-py raises `ValueError: negative shift count` — an incidental failure from
  `1 << (59 - 2 * parent_resolution)` going negative, not a deliberate check.
- a5-rs returns a garbage cell ID, because the equivalent shift wraps.

Failing loudly beats returning nonsense, but neither is a contract. a5-rs should
validate `parent_resolution <= get_resolution(index)` and return
`Err(...)`; a5-py and a5-js should raise the same message.

**Pinned by:** `tests/test_differential.py::test_known_divergences`, and worked
around in `TestHierarchy::test_cell_to_parent`, which bounds its sweep by the
cell's actual resolution.

---

## 4. All ports: `lonlat_to_cell` at the exact poles depends on call history

**Severity:** needs investigation. Narrow trigger (exactly `lat == ±90.0`), but
the underlying property — results depending on previous calls — is serious.

From a cold interpreter both ports agree and both get the right hemisphere:

```
lonlat_to_cell((0, -90), 10)   ->  13258597852734554112   lat -89.957   (both ports)
lonlat_to_cell((0, +90), 10)   ->   1273017311418122240   lat +89.957   (both ports)
```

But `spherical_to_cell` keeps a `_last_result` origin hint — it re-projects
against the previous call's face and returns early if the point falls inside
that pentagon. At a pole, which face you project against decides the answer, so
**the result depends on what was called before**. Reproduction, each line a
fresh interpreter:

```sh
# north pole first, then south:
A5_BACKEND=rust   python -c "import a5; a5.lonlat_to_cell((0,90),10); \
    print(a5.cell_to_lonlat(a5.lonlat_to_cell((0,-90),10)))"
#   -> (-71.61, +89.957)      wrong hemisphere, ~20000 km out, silently

A5_BACKEND=python python -c "import a5; a5.lonlat_to_cell((0,90),10); \
    a5.lonlat_to_cell((0,-90),10)"
#   -> ZeroDivisionError
```

So in that order a5-rs answers confidently and wrongly while a5-py raises —
a5-py is the better-behaved of the two, though neither is right. Calling
`south, north, south` in one process gives the *correct* southern cell on the
third call, so the state dependence is not a simple "previous call wins"; it has
not been fully characterised.

**Root cause (partial):** `rho = (D / volume_abc) * sqrt(...)` in
`equal_area.rs` / `equal_area.py`, where `volume_abc` is zero when the point is
projected against a face it lies on the vertex of. Rust and JavaScript produce
`inf` and continue; Python raises. The TypeScript source has the same unguarded
division in `modules/projections/equal-area.ts`, so a5-js is likely to behave
like a5-rs — worth confirming.

**Fix:** two separable pieces, and the second is the one that matters:

1. Guard the degenerate projection. Needs a decision about what the poles *should*
   map to — a pole is a vertex shared between faces, so the answer is a
   convention, not a derivation.
2. Make `lonlat_to_cell` a pure function of its arguments. The origin hint is a
   performance cache and must not be able to change results; today it can, in
   every port.

**Not pinned by a test.** A test would have to control global cache state across
all three ports to be deterministic, and asserting today's broken-and-not-fully-
understood behaviour would be worse than no test. The generated corpus samples
`asin(uniform(-1, 1))` and never lands exactly on a pole, so nothing catches
this automatically — reproduce with the commands above.
