# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

# The turtle alphabet on the integer (a,b) triangular lattice (basis u=(sqrt3/4,1/4),
# v=(0,1/2)). Draw symbols {E e S s U u D d T t} are unit segments; `+`/`-` are 60 deg
# turns. Each symbol also carries the 3 corners of the triangular cell it hosts.
# Everything here is exact integer -- sqrt3 only enters when (a,b) is later mapped to
# A5 triple coordinates (in the lsystem package).

from typing import Callable, List, NamedTuple, Optional, Tuple


class AB(NamedTuple):
    """A point on the integer (a,b) triangular lattice."""
    a: int
    b: int


def add(p: AB, q: AB) -> AB:
    return AB(p.a + q.a, p.b + q.b)


def neg(p: AB) -> AB:
    """180 deg rotation."""
    return AB(-p.a, -p.b)


def _rot60(p: AB) -> AB:
    """60 deg CCW, order 6."""
    return AB(-p.b, p.a + p.b)


def rot_times(p: AB, n: int) -> AB:
    r = p
    k = n % 6
    for _ in range(k):
        r = _rot60(r)
    return r


# Step vector of each draw symbol at heading 0. Lowercase = same step, cell hosted
# on the other side (see _HOST_OFFSETS).
_BASE = {
    'E': AB(4, 0), 'e': AB(4, 0),
    'S': AB(4, -2), 's': AB(4, -2),
    'U': AB(0, 2), 'u': AB(0, 2),
    'D': AB(0, -2), 'd': AB(0, -2),
    'T': AB(-4, 0), 't': AB(-4, 0),
}

DRAW = set(_BASE.keys())


def is_draw(sym: str) -> bool:
    """Whether a symbol is a draw terminal."""
    return sym in _BASE


# The 3 corner offsets (heading 0, from the segment start) of the cell each symbol hosts.
_HOST_OFFSETS = {
    'E': (AB(0, 0), AB(4, 0), AB(4, -4)),
    'e': (AB(0, 0), AB(4, 0), AB(0, 4)),
    'S': (AB(0, 0), AB(4, 0), AB(4, -4)),
    's': (AB(4, -2), AB(0, 2), AB(0, -2)),
    'U': (AB(0, 2), AB(0, -2), AB(4, -2)),
    'u': (AB(0, 0), AB(0, 4), AB(-4, 4)),
    'D': (AB(0, 2), AB(0, -2), AB(4, -2)),
    'd': (AB(0, 0), AB(0, -4), AB(-4, 0)),
    'T': (AB(0, -4), AB(-4, 0), AB(-4, -4)),
    't': (AB(-4, 4), AB(0, 0), AB(0, 4)),
}


def host_corners(sym: str, frm: AB, heading: int) -> Tuple[AB, AB, AB]:
    """The 3 (a,b) corners of the cell hosted by `sym`, drawn from `frm` at `heading`."""
    o = _HOST_OFFSETS[sym]
    return (
        add(frm, rot_times(o[0], heading)),
        add(frm, rot_times(o[1], heading)),
        add(frm, rot_times(o[2], heading)),
    )


def host_sum(sym: str, frm: AB, heading: int) -> AB:
    """The corner SUM (= 3*centroid, an exact integer) of that cell."""
    p, q, r = host_corners(sym, frm, heading)
    return AB(p.a + q.a + r.a, p.b + q.b + r.b)


class TurtleState(NamedTuple):
    pos: AB
    heading: int


def walk(
    s: str,
    pos: AB,
    heading: int,
    on_draw: Optional[Callable[[str, AB, int], None]] = None,
) -> TurtleState:
    """
    Walk a draw string (draw symbols + `+`/`-` turns) from (pos, heading). Calls
    `on_draw(sym, frm, heading)` for each draw symbol (before advancing). Returns the
    final turtle state.
    """
    p = pos
    h = heading % 6
    for ch in s:
        if ch == '+':
            h = (h + 1) % 6
            continue
        if ch == '-':
            h = (h + 5) % 6
            continue
        step = _BASE.get(ch)
        if step is None:
            continue
        if on_draw is not None:
            on_draw(ch, p, h)
        p = add(p, rot_times(step, h))
    return TurtleState(pos=p, heading=h)


def net_of(s: str) -> Tuple[AB, int]:
    """Net (a,b) displacement + net heading of a draw string, from origin at heading 0."""
    end = walk(s, AB(0, 0), 0)
    return end.pos, end.heading
