# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

# The A5 curve's L-system grammar: the motif rules and the string operations that
# expand them. This file is the source of truth for the curve's definition and is
# purely symbolic -- the turtle geometry that interprets the symbols lives in
# turtle.py, and the compilation to descent tables in tables.py.
#
# 7 self-referential motifs over the alphabet {A B C M P Q R} (+ their lowercase
# reverses) + the draw terminals {E e S s U u D d T t} + the 60 deg turns +/-.
# A LOWERCASE motif is its uppercase counterpart REVERSED, generated automatically
# by `reverse_motif` -- so only the 7 uppercase rules below need to be authored.

from typing import Dict, List

# Each motif's production rule (the 7 authored motifs).
RULES: Dict[str, str] = {
    'A': 'PQAB',
    'B': 'B+++PQ---A',
    'C': 'P---RMb+++',
    'M': 'qQ+++C---b',
    'P': 'PpB---B+++',
    'Q': 'PQ---Cb+++',
    'R': 'b+++a---qQ',
}

# Each motif's leaf draw symbol -- the terminal it renders as at the base case.
DRAWS: Dict[str, str] = {
    'A': 'E', 'B': '+e-', 'C': '-e+', 'M': 'T', 'P': 'S', 'Q': 'D', 'R': '+++D---',
}

# The authored (uppercase) motif keys.
MOTIFS: List[str] = list(RULES.keys())

# All motif keys, uppercase + their lowercase (reversed) counterparts.
ALL_MOTIFS: List[str] = MOTIFS + [m.lower() for m in MOTIFS]


def _swap_case(c: str) -> str:
    return c.upper() if c.islower() else c.lower()


def reverse_motif(s: str) -> str:
    """
    The reverse of a motif/draw string -- traced end to start. Uniform transform:
    reverse the order, swap the case of every letter (uppercase<->lowercase =
    forward<->reverse partner), and flip every `+`/`-`. This is how the lowercase
    motifs are derived from the authored uppercase rules.
    """
    out = []
    for c in reversed(s):
        if c == '+':
            out.append('-')
        elif c == '-':
            out.append('+')
        else:
            out.append(_swap_case(c))
    return ''.join(out)


def expand_once(s: str, table: Dict[str, str]) -> str:
    """
    One expansion pass over `s`: replace each symbol using `table` (RULES or
    DRAWS). A lowercase motif whose uppercase is in `table` expands to that rule
    REVERSED; turns and unknown symbols pass through unchanged.
    """
    out = []
    for ch in s:
        up = ch.upper()
        if ch in table:
            out.append(table[ch])
        elif ch != up and up in table:
            out.append(reverse_motif(table[up]))
        else:
            out.append(ch)
    return ''.join(out)
