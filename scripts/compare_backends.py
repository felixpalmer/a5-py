#!/usr/bin/env python3
# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors
#
# Reports how much faster the compiled backend is than pure Python, from two
# pytest-benchmark JSON runs of the same suite.
#
# Usage:
#   python scripts/compare_backends.py <python.json> <rust.json>
#
# Unlike scripts/compare_benchmarks.py this gates nothing -- it is a report. The
# regression gate compares pure Python against pure Python across commits, where
# a threshold is meaningful; the backend ratio depends on the machine and on how
# much of each benchmark is interpreter overhead, so there is no sensible fixed
# bound to enforce.
#
# Comparison keys off each benchmark's MINIMUM sample time (stats.min) for the
# same reason as compare_benchmarks.py: it is the least environment-perturbed
# sample and is stable to a few percent between runs on shared CI hardware.

import json
import sys


def format_time(seconds):
    ns = seconds * 1e9
    if ns < 1e3:
        return f"{ns:.1f}ns"
    if ns < 1e6:
        return f"{ns / 1e3:.2f}µs"
    if ns < 1e9:
        return f"{ns / 1e6:.2f}ms"
    return f"{seconds:.2f}s"


def format_ratio(ratio):
    if ratio >= 100:
        return f"{ratio:.0f}x"
    if ratio >= 10:
        return f"{ratio:.1f}x"
    return f"{ratio:.2f}x"


def load(path):
    data = json.load(open(path))["benchmarks"]
    # Key on fullname (file::function) so matching is stable across runs.
    return {b["fullname"]: b for b in data}


def median(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def render_table(rows):
    lines = [
        "| benchmark | pure python | rust | speedup |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(f"| {r['name']} | {r['python']} | {r['rust']} | {r['speedup']} |")
    return lines


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print(
            "Usage: python scripts/compare_backends.py <python.json> <rust.json>",
            file=sys.stderr,
        )
        sys.exit(2)

    python_runs = load(args[0])
    rust_runs = load(args[1])

    def time_of(b):
        return b["stats"]["min"]

    rows = []
    ratios = []
    unmatched = []

    for fullname, rust_bench in rust_runs.items():
        python_bench = python_runs.get(fullname)
        if python_bench is None:
            unmatched.append(rust_bench.get("name", fullname))
            continue
        rust_time = time_of(rust_bench)
        python_time = time_of(python_bench)
        ratio = python_time / rust_time if rust_time > 0 else 0.0
        ratios.append(ratio)
        rows.append(
            {
                "name": rust_bench.get("name", fullname),
                "python": format_time(python_time),
                "rust": format_time(rust_time),
                "speedup": format_ratio(ratio),
                "ratio": ratio,
            }
        )

    rows.sort(key=lambda r: r["ratio"], reverse=True)

    lines = ["## Backend comparison", ""]
    if not ratios:
        lines.append("No benchmarks matched between the two runs.")
        print("\n".join(lines))
        sys.exit(1)

    lines.append(
        "_Pure-Python time divided by compiled-backend time, on the minimum sample "
        "per benchmark. Higher is better for the compiled backend._"
    )
    lines.append("")
    lines.append(
        "_Benchmarks that call internal modules directly rather than the `a5` "
        "public API -- bench_projections.py and bench_curve.py -- run pure Python "
        "on both sides and sit at 1.00x by construction._"
    )
    lines.append("")
    lines.append(
        f"**Median {format_ratio(median(ratios))}** across {len(ratios)} benchmarks "
        f"(range {format_ratio(min(ratios))} – {format_ratio(max(ratios))})."
    )
    lines.append("")

    slower = [r for r in rows if r["ratio"] < 1.0]
    if slower:
        lines.append(
            f"### ⚠️ {len(slower)} benchmark(s) where the compiled backend is slower"
        )
        lines.append("")
        lines += render_table(slower)
        lines.append("")

    if unmatched:
        lines.append(
            f"_{len(unmatched)} benchmark(s) present only in the compiled run "
            "(not compared)._"
        )
        lines.append("")

    lines.append("<details>")
    lines.append(f"<summary>All results ({len(rows)} benchmarks)</summary>")
    lines.append("")
    lines += render_table(rows)
    lines.append("")
    lines.append("</details>")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
