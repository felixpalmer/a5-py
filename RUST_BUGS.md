# Cross-port divergences found by the differential suite

`tests/test_differential.py` runs the pure-Python and compiled backends over the
same generated corpus and compares them. Everything below is a case where they
disagree. Each is pinned by a test so it cannot drift further, and each needs
fixing in its home repository rather than being papered over in `a5/_native.py`.

Pinned against a5-rs `7ad0978` (0.10.0), the revision `Cargo.toml` names.

---

## 1. a5-rs: `get_num_cells` returns wrong values for resolutions 28-30

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
Both fail when upstream is fixed and the pin is bumped, which is the signal to
delete the markers.

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

## 4. a5-py: `lonlat_to_cell` divides by zero at the south pole

**Severity:** low; exact-pole input only.

`lonlat_to_cell((lon, -90.0), resolution)` raises `ZeroDivisionError` from
`a5/projections/equal_area.py`, where `volume_abc` is zero. a5-rs returns a cell
for the same input. The north pole works in both.

Here a5-rs has the better behaviour and a5-py needs the fix. Note also that at
the exact north pole *both* ports return a result that depends on the
origin-hint cache — a first call gives a different cell from a warm one. That
quirk is mirrored faithfully across ports, so it is not a divergence, but it is
its own latent bug worth a separate look.

**Not currently pinned:** the differential corpus samples
`asin(uniform(-1, 1))`, which never lands exactly on a pole.
