# Cross-port divergences found by the differential suite

`tests/test_differential.py` runs the pure-Python and compiled backends over the
same generated corpus and compares them. Everything below is a case where they
disagree. Each needs fixing in its home repository rather than being papered
over in `a5/_native.py`.

Observed against a5-rs `4091ab6` (1.0.0-beta.2), the revision `Cargo.toml` names.
Entries keep their original numbers; #1 (`get_num_cells` wrong at resolutions
28-30) was fixed upstream and removed.

| # | what | where the fix belongs | status | pinned by a test |
| --- | --- | --- | --- | --- |
| 2 | `hex_to_u64` accepts too much | a5-py (+ a decision for a5-js) | open | yes |
| 3 | `cell_to_parent` unchecked above the cell's resolution | a5-rs, a5-py, a5-js | open | no |
| 4 | pole results depend on call history | all ports | open, not characterised | no |

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
Neither port rejects it: since the v1 bit-op rewrite of `cell_to_parent`, both
return the same garbage cell ID (e.g. a res-8 cell asked for its res-12
"parent"), so the ports agree but the answer is nonsense.

Failing loudly beats returning nonsense. a5-rs should
validate `parent_resolution <= get_resolution(index)` and return
`Err(...)`; a5-py and a5-js should raise the same message.

**Not pinned**, since the ports now agree. `TestHierarchy::test_cell_to_parent`
bounds its sweep by the cell's actual resolution to stay in domain.

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
