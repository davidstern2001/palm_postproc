#!/usr/bin/env python3
"""
diag_join_times.py
------------------
Inspects the time axes of the part files that palm_postproc's join step
stitches together, and reports where they overlap, interleave or disagree.

Run this against the raw OUTPUT directory when join fails with
    IndexError: size of data array does not conform to slice
It prints the per-part time ranges, the global timestep union, and — for
each part — the span it occupies in that union. A part whose global span
is wider than its own timestep count is the one that broke the join.

Read-only. Writes nothing.
"""

import glob
import os
import re
import sys

import numpy as np
from netCDF4 import Dataset

# ------------------------------
# 1. INPUT AND OUTPUT FILES
# ------------------------------

OUTDIR = "/home/stern/palm/model.git/build/JOBS/holesovice_stern_adapt/OUTPUT"
SUFFIX = ".nc"

# File stems to inspect. Empty list = auto-detect every stem in OUTDIR.
STEMS = [  # e.g. "holesovice_stern_adapt_av_3d"

]

# Only report a spatial-dimension mismatch for these dimensions.
CHECK_DIMS = ("x", "y", "xu", "yv", "zu_3d", "zw_3d", "zu_xy", "zw_xy")


# ------------------------------
# 2. HELPERS
# ------------------------------

def detect_stems(outdir, suffix):
    """Same auto-detection rule as join._detect_filelist."""
    stems = set()
    for p in glob.glob(os.path.join(outdir, "*" + suffix)):
        name = os.path.basename(p)
        stem = name[: -len(suffix)] if suffix else name
        stems.add(re.sub(r"\.\d{3}$", "", stem))
    return sorted(stems)


def part_files(outdir, stem, suffix):
    """Same 'filenum' convention as join._partnames."""
    return sorted(glob.glob(os.path.join(outdir, stem + ".*" + suffix)))


def read_part(path):
    """Return (times, dims, tvars) for one part file."""
    with Dataset(path, "r") as ds:
        dims = {k: len(v) for k, v in ds.dimensions.items()}
        tvars = sorted(v for v in ds.variables
                       if ds.variables[v].dimensions[:1] == ("time",))
        if "time" not in ds.variables:
            return None, dims, tvars
        raw = ds.variables["time"][:]
        good = raw[~np.ma.getmaskarray(raw)] if np.ma.isMA(raw) else raw
        return np.asarray(good, dtype=float), dims, tvars


# ------------------------------
# 3. INSPECT ONE STEM
# ------------------------------

def inspect(stem, outdir, suffix):
    print("=" * 74)
    print(stem)
    print("=" * 74)

    parts = part_files(outdir, stem, suffix)
    if not parts:
        print("  no part files found")
        return 0

    print(f"  {len(parts)} part file(s)")
    print()

    info = []
    for path in parts:
        name = os.path.basename(path)
        try:
            times, dims, tvars = read_part(path)
        except Exception as exc:
            print(f"  {name:<48s} UNREADABLE: {exc}")
            continue
        if times is None:
            print(f"  {name:<48s} static (no time variable)")
            continue
        if times.size == 0:
            print(f"  {name:<48s} 0 valid timesteps (truncated / empty)")
            continue
        step = np.median(np.diff(times)) if times.size > 1 else float("nan")
        print(f"  {name:<48s} {times.size:>4d} steps  "
              f"t={times[0]:.1f} … {times[-1]:.1f}  dt≈{step:.1f}")
        info.append({"name": name, "times": times, "dims": dims,
                     "tvars": tvars})

    if not info:
        print("\n  nothing to join")
        return 0

    # ------------------------------
    # 4. SPATIAL DIMENSION AGREEMENT
    # ------------------------------
    print("\n  spatial dimensions")
    ref = info[0]
    bad_dims = 0
    for pi in info:
        diff = {d: (pi["dims"].get(d), ref["dims"].get(d))
                for d in CHECK_DIMS
                if d in ref["dims"] and pi["dims"].get(d) != ref["dims"][d]}
        if diff:
            bad_dims += 1
            for d, (got, want) in diff.items():
                print(f"    {pi['name']}: {d} = {got}, "
                      f"{ref['name']} has {want}")
        missing = sorted(set(ref["tvars"]) - set(pi["tvars"]))
        extra = sorted(set(pi["tvars"]) - set(ref["tvars"]))
        if missing or extra:
            bad_dims += 1
            print(f"    {pi['name']}: variable set differs "
                  f"(missing {missing}, extra {extra})")
    if not bad_dims:
        print("    all parts agree")

    # ------------------------------
    # 5. GLOBAL TIMESTEP UNION
    # ------------------------------
    union = sorted({float(t) for pi in info for t in pi["times"]})
    arr = np.asarray(union)
    diffs = np.diff(arr)
    diffs = diffs[diffs > 0]
    tol = float(np.median(diffs)) * 0.25 if diffs.size else 1e-6

    merged = []
    for t in union:
        if merged and abs(t - merged[-1]) < tol:
            merged[-1] = t
        else:
            merged.append(t)

    print(f"\n  global timesteps: {len(merged)}  "
          f"(union {len(union)}, merge tolerance {tol:.4g} s)")
    if len(merged) < len(union):
        print(f"    {len(union) - len(merged)} near-duplicate(s) merged "
              f"— restart-cycle boundaries")

    # ------------------------------
    # 6. PART → GLOBAL SPAN
    # ------------------------------
    print("\n  part span in the global timestep list")
    garr = np.asarray(merged)
    broken = 0
    for pi in info:
        idx = []
        for t in pi["times"]:
            k = int(np.argmin(np.abs(garr - t)))
            if abs(garr[k] - t) < tol:
                idx.append(k)
        if not idx:
            print(f"    {pi['name']:<46s} contributes nothing")
            continue
        span = max(idx) - min(idx) + 1
        flag = ""
        if span != len(idx):
            broken += 1
            flag = "   <-- INTERLEAVED, this breaks the old join"
        print(f"    {pi['name']:<46s} global[{min(idx)}…{max(idx)}] "
              f"span={span} own={len(idx)}{flag}")

    print()
    if broken:
        print(f"  RESULT: {broken} part(s) interleave with another part.")
        print("          This is the 'size of data array does not conform "
              "to slice' cause.")
    elif bad_dims:
        print("  RESULT: time axes are fine; the parts disagree on shape "
              "or variables.")
    else:
        print("  RESULT: nothing wrong found in the time axes.")
    print()
    return broken


# ------------------------------
# 7. MAIN
# ------------------------------

def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else OUTDIR
    if not os.path.isdir(outdir):
        print(f"not a directory: {outdir}")
        return 2

    stems = STEMS or detect_stems(outdir, SUFFIX)
    print(f"input dir: {outdir}")
    print(f"stems:     {len(stems)}\n")

    broken = 0
    for stem in stems:
        broken += inspect(stem, outdir, SUFFIX)
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
