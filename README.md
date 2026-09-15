# [A5](https://a5geo.org) - Global, equal-area, millimeter-accurate geospatial index

## Website: [A5Geo.org](https://a5geo.org)

<img width="472" alt="thumbnails" src="https://github.com/user-attachments/assets/865d3afa-a4d6-4e86-b814-252587b10ca0" />

## Introduction
  
A5 is a geospatial index that partitions the world into [pentagonal cells](https://a5geo.org/examples/teohedron-dodecahedron). The cells are available at [31 different resolution levels](https://a5geo.org/examples/hierarchy), with the largest cell covering the whole world, and the smallest less than 30mm². Within each resolution level the cells have [equal area](https://a5geo.org/examples/area), as per the [OGC definition](https://docs.ogc.org/as/20-040r3/20-040r3.html#toc29).

A5 cells provide simple way to represent spatial data as a collection of cells, which together represent regions on the globe. These can be anything, from city districts to parcels of land. Once data is in a cell-based format, it becomes much simpler to perform analysis such as calculating the correlation between different variables - for example elevation and crop yield.

Cells can also be used to group (aggregate) point data, to understand how it spatially distributed. For example, the [density of holiday rentals](https://a5geo.org/examples/airbnb) across a city.

To understand how it works, take a look at the [Examples](https://a5geo.org/examples).

A5 is implemented in TypeScript and is available as a [library](https://www.npmjs.com/package/a5-js), with [API documentation here](https://a5geo.org/docs/api-reference/). It is [open source](https://github.com/felixpalmer/a5) and licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0.txt).

## Installation

```bash
pip install pya5
```

## Backends

`pya5` ships two interchangeable implementations of the same public API:

| backend | what it is | when it runs |
| --- | --- | --- |
| `python` | the pure-Python implementation | the default in 0.x |
| `rust` | PyO3 bindings to the [a5-rs](https://github.com/felixpalmer/a5-rs) crate, roughly two orders of magnitude faster | opt in with `A5_BACKEND=rust` |

Signatures, return values and exceptions are identical either way, so nothing in
your code changes when you switch. Select with the `A5_BACKEND` environment
variable, read once at import time:

```bash
A5_BACKEND=rust    # require the compiled backend; raises ImportError if absent
A5_BACKEND=auto    # use it when available, fall back to pure Python silently
A5_BACKEND=python  # always pure Python
```

`A5_PURE_PYTHON=1` is accepted as an alias for `A5_BACKEND=python`. Check which
one is live with `a5.get_backend()`.

**0.x defaults to `python`** so that upgrading cannot change results or timings
underneath you; `rust` is opt-in while it gets real-world exposure. In 1.0 the
default becomes `auto`, making the compiled backend standard wherever a wheel
provides one.

### Batch operations

`cell_to_parent`, `cell_to_children`, `get_resolution` and `cell_area` each do so
little work that, called one at a time, the Python/Rust boundary crossing costs
more than the operation. `a5.batch` takes a sequence and returns a list, paying
that cost once per batch instead of once per element:

```python
from a5 import batch

parents = batch.cell_to_parent(cells, 5)
resolutions = batch.get_resolution(cells)
```

Both backends implement these, so they work regardless of which one is selected.

### Building the compiled backend from source

Platform wheels bundle it already. To build it in a working copy you need a
[Rust toolchain](https://rustup.rs):

```bash
uv run maturin develop --release
```

## Benefits over alternative systems

Like other [DGGSs](https://a5geo.org/docs/technical/dggs), A5 can be used for indexing, spatial joins, and other spatial operations. The appropriate choice of DGGS will depend on the use case, the key strengths of A5 over other similar indexing systems are:

- Equal area cells group spatial features without introducing bias.
- Very high resolution of 30mm² at final resolution level, encoded as a 64-bit integer. 
- Single cell topology type, with minimal shape distortion across the globe.

## Geometric Construction

[DGGSs](https://a5geo.org/docs/technical/dggs) are generally based on a planar cell tiling which is applied to the sides of a [platonic solid](https://a5geo.org/docs/technical/platonic-solids), before being projected onto a sphere.

A5 is unique in that it uses a pentagonal tiling of a dodecahedron. The [pentagon chosen](https://a5geo.org/docs/technical/the-pentagon-that-could) for the tiling is equilateral, but not regular - unlike other common DGGSs which use regular polygons, for example [HTM](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/09/tr-2005-123.pdf) uses triangles, [S2](https://s2geometry.io/) uses squares, and [H3](https://h3geo.org/) uses hexagons.

The benefit of choosing a dodecahedron is that it is the platonic solid with the lowest vertex curvature, and by this measure it is the most spherical of all the platonic solids. This is key for minimizing cell distortion as the process of projecting a platonic solid onto a sphere involves warping the cell geometry to force the vertex curvature to approach zero. Thus, the lower the original vertex curvature, the less distortion will be introduced by the projection.
