#!/usr/bin/env python3
# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors
#
# Compares two pytest-benchmark JSON runs (produced with --benchmark-json) and
# exits non-zero if any benchmark regressed by more than the threshold.
#
# Usage:
#   python scripts/compare_benchmarks.py <baseline.json> <current.json> [threshold%]
#
# Comparison keys off each benchmark's MINIMUM sample time (stats.min), not its
# mean. The minimum is the least environment-perturbed sample (no GC pause or
# scheduler preemption landing mid-measurement), so it is far more stable
# between runs on shared CI hardware. Means of allocation-heavy benchmarks swing
# widely run-to-run while their minimums agree within a few percent.
#
# Output is GitHub-flavored markdown for $GITHUB_STEP_SUMMARY: regressions and
# gains beyond the threshold are surfaced in their own tables at the top, with
# the full results in a collapsed <details> section below.

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


def format_delta(delta):
    return f"{'+' if delta >= 0 else ''}{delta:.1f}%"


def render_table(rows):
    lines = ["| benchmark | baseline | current | change |", "| --- | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['baseline']} | {r['current']} | {r['change']} |")
    return lines


def load(path):
    data = json.load(open(path))["benchmarks"]
    # Key on fullname (file::function) so matching is stable across runs.
    return {b["fullname"]: b for b in data}


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print("Usage: python scripts/compare_benchmarks.py <baseline.json> <current.json> [threshold%]", file=sys.stderr)
        sys.exit(2)
    baseline_path, current_path = args[0], args[1]
    threshold = float(args[2]) if len(args) > 2 else 15.0

    baseline = load(baseline_path)
    current = load(current_path)

    # Compare on minimum sample time.
    def time_of(b):
        return b["stats"]["min"]

    rows = []
    regressions = []
    gains = []
    added = 0

    for fullname, bench in current.items():
        display = bench.get("name", fullname)
        base = baseline.pop(fullname, None)
        if base is None:
            added += 1
            rows.append({"name": display, "baseline": "—", "current": format_time(time_of(bench)), "change": "new"})
            continue
        delta = 100.0 * (time_of(bench) - time_of(base)) / time_of(base)
        row = {
            "name": display,
            "baseline": format_time(time_of(base)),
            "current": format_time(time_of(bench)),
            "change": format_delta(delta),
            "delta": delta,
        }
        rows.append(row)
        if delta > threshold:
            regressions.append(row)
        elif delta < -threshold:
            gains.append(row)

    removed = list(baseline.values())
    for b in removed:
        rows.append({"name": b.get("name", b["fullname"]), "baseline": format_time(time_of(b)), "current": "—", "change": "removed"})

    regressions.sort(key=lambda r: r["delta"], reverse=True)
    gains.sort(key=lambda r: r["delta"])

    lines = ["## Benchmark comparison", ""]
    lines.append("_Times are the minimum sample per benchmark (most stable metric across runs)._")
    lines.append("")

    if regressions:
        lines.append(f"### ❌ {len(regressions)} regression{'' if len(regressions) == 1 else 's'} above {threshold:g}%")
        lines.append("")
        lines += render_table([{**r, "change": f"**{r['change']}**"} for r in regressions])
        lines.append("")
    else:
        lines.append(f"### ✅ No regressions above {threshold:g}%")
        lines.append("")

    if gains:
        lines.append(f"### \U0001f680 {len(gains)} gain{'' if len(gains) == 1 else 's'} above {threshold:g}%")
        lines.append("")
        lines += render_table([{**r, "change": f"**{r['change']}**"} for r in gains])
        lines.append("")

    if added or removed:
        lines.append(f"_{added} benchmark(s) added, {len(removed)} removed (not compared)._")
        lines.append("")

    lines.append("<details>")
    lines.append(f"<summary>All results ({len(rows)} benchmarks)</summary>")
    lines.append("")
    lines += render_table(rows)
    lines.append("")
    lines.append("</details>")

    print("\n".join(lines))
    sys.exit(1 if regressions else 0)


if __name__ == "__main__":
    main()
