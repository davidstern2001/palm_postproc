#!/usr/bin/env python3
"""
test.py -- palm_postproc self-test
==================================

Builds a small synthetic PALM output tree in a temporary directory, runs the
pipeline over it in every mode, and verifies the results. No real data and no
PALM installation needed; the whole thing takes a few seconds.

    python test.py            # run everything
    python test.py --keep     # keep the temporary tree and print its path

What it covers
--------------
  * dependency check with a clear message per missing package
  * join           : two part files stitched into one, timestep union
  * splitvar       : one file per variable, u/v/w grouped, '*' stripped
  * splitz         : z <= z_max, on EVERY vertical coordinate (zu_3d AND
                     zw_3d, which is what the .uvw group file carries)
  * splittime      : subsampling by interval
  * coord          : UTM offsets, WGS84 aux coords, CF epoch handling,
                     kelvin -> degrees C on all four temperature prefixes,
                     fill values NOT converted
  * chain mode, classic (chain: false) mode and --only, since the classic
    path is the one that used to crash on a missing cfg.paths.input
  * resume: a second run skips everything, and touching the source redoes it
  * provenance attributes on the written files

Exit code is 0 when every check passes, 1 otherwise.
"""

import argparse
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

# ------------------------------
# 1. INPUT AND OUTPUT SETTINGS
# ------------------------------
CASE = "selftest"
NX, NY, NZ = 6, 5, 8
DZ = 10.0
ORIGIN_X, ORIGIN_Y = 458000.0, 5548000.0
ORIGIN_TIME = "2023-08-23 12:00:00 +02"
EPSG = 32633

# two parts, overlapping in time the way PALM restart chains do
PART_TIMES = ([0.0, 600.0, 1200.0], [1200.0, 1800.0, 2400.0])
FILL = -9999.0


# ------------------------------
# 2. DEPENDENCY CHECK
# ------------------------------
def check_dependencies():
    print("Checking dependencies ...")
    required = {
        "numpy":   "numpy",
        "xarray":  "xarray",
        "netCDF4": "netCDF4",
        "yaml":    "pyyaml",
        "pyproj":  "pyproj",
    }
    missing = []
    for mod, pkg in required.items():
        try:
            __import__(mod)
            print(f"  ok      {mod}")
        except ImportError:
            print(f"  MISSING {mod}   (pip install {pkg})")
            missing.append(pkg)
    try:
        __import__("dask")
        print("  ok      dask (optional, enables chunked reads)")
    except ImportError:
        print("  --      dask not installed (optional; large files will be "
              "read whole)")
    if missing:
        print("\nInstall the missing package(s) and run again:")
        print("  pip install " + " ".join(missing))
        return False
    return True


# ------------------------------
# 3. SYNTHETIC PALM OUTPUT
# ------------------------------
def _make_part(path, times, mode):
    """One PALM output part file: _av_xy (2D) or _av_3d (3D)."""
    import numpy as np
    from netCDF4 import Dataset

    is_3d = (mode == "3d")
    with Dataset(path, "w", format="NETCDF4") as nc:
        nc.origin_x = ORIGIN_X
        nc.origin_y = ORIGIN_Y
        nc.origin_z = 0.0
        nc.origin_time = ORIGIN_TIME
        nc.dt_averaging = 600.0

        nc.createDimension("time", None)
        nc.createDimension("x", NX)
        nc.createDimension("y", NY)
        nc.createDimension("xu", NX)
        nc.createDimension("yv", NY)
        if is_3d:
            nc.createDimension("zu_3d", NZ)
            nc.createDimension("zw_3d", NZ)

        tv = nc.createVariable("time", "f8", ("time",))
        # A CF epoch DIFFERENT from origin_time, so the coord step's epoch
        # shift is actually exercised (a restarted run looks like this).
        tv.units = "seconds since 2023-08-23 10:00:00"
        tv.calendar = "standard"
        tv[:] = np.asarray(times) + 7200.0     # +2 h back to origin_time

        xv = nc.createVariable("x", "f8", ("x",))
        xv[:] = (np.arange(NX) + 0.5) * DZ
        xv.units = "m"
        yv = nc.createVariable("y", "f8", ("y",))
        yv[:] = (np.arange(NY) + 0.5) * DZ
        yv.units = "m"
        nc.createVariable("xu", "f8", ("xu",))[:] = np.arange(NX) * DZ
        nc.createVariable("yv", "f8", ("yv",))[:] = np.arange(NY) * DZ

        if is_3d:
            nc.createVariable("zu_3d", "f8", ("zu_3d",))[:] = \
                (np.arange(NZ) + 0.5) * DZ
            nc.createVariable("zw_3d", "f8", ("zw_3d",))[:] = \
                np.arange(NZ) * DZ

        nt = len(times)
        if is_3d:
            dims3 = ("time", "zu_3d", "y", "x")
            shape3 = (nt, NZ, NY, NX)
            for name, base in (("u", 1.0), ("v", 2.0), ("theta", 290.0)):
                v = nc.createVariable(name, "f4", dims3, fill_value=FILL)
                v.units = "K" if name == "theta" else "m/s"
                v[:] = base + np.zeros(shape3, dtype="f4")
            # w lives on the STAGGERED zw_3d: the reason splitz must slice
            # every vertical coordinate and not just the configured one.
            w = nc.createVariable("w", "f4", ("time", "zw_3d", "y", "x"),
                                  fill_value=FILL)
            w.units = "m/s"
            w[:] = 0.5
            ta = nc.createVariable("ta", "f4", dims3, fill_value=FILL)
            ta.units = "K"
            ta[:] = 288.0
        else:
            dims2 = ("time", "y", "x")
            shape2 = (nt, NY, NX)
            for name, base, unit in (("theta_2m*_xy", 295.0, "K"),
                                     ("ta_2m*_xy", 293.0, "K"),
                                     ("t_surf*_xy", 300.0, "K"),
                                     ("wspeed_xy", 3.0, "m/s")):
                v = nc.createVariable(name, "f4", dims2, fill_value=FILL)
                v.units = unit
                arr = base + np.zeros(shape2, dtype="f4")
                arr[:, 0, 0] = FILL          # a NoData cell to protect
                v[:] = arr
            # time-invariant variable: must not gain a bogus time dimension
            zi = nc.createVariable("ind_z_xy", "f4", ("y", "x"))
            zi[:] = 1.0


def build_dataset(root):
    """Create <root>/<CASE>/OUTPUT with two parts of each file type."""
    out = Path(root) / CASE / "OUTPUT"
    out.mkdir(parents=True, exist_ok=True)
    for mode, stem in (("xy", f"{CASE}_av_xy_N02"),
                       ("3d", f"{CASE}_av_3d_N02")):
        for ipart, times in enumerate(PART_TIMES):
            _make_part(out / f"{stem}.{ipart:03d}.nc", times, mode)
    return out


def make_config(root, path, extra=""):
    """Write a config YAML and return its path."""
    text = f"""
case: {CASE}
paths:
  base: {root}
complevel: 1
workers: 1
chain: true
steps:
  join:
    enabled: true
    convention: filenum
  splitvar:
    enabled: true
    vars_2d: all
    vars_3d: all
  splitz:
    enabled: true
    z_max: 45.0
    z_coord: zu_3d
  splittime:
    enabled: true
    timestep: "20m"
  coord:
    enabled: true
    crs: both
    utm_zone: {EPSG}
    celsius: true
{extra}
"""
    Path(path).write_text(text.lstrip())
    return Path(path)


# ------------------------------
# 4. CHECK HELPERS
# ------------------------------
def check(cond, msg, failures):
    if cond:
        print(f"  ok      {msg}")
    else:
        print(f"  FAIL    {msg}")
        failures.append(msg)
    return bool(cond)


# ------------------------------
# 5. UNIT CHECKS (pure functions)
# ------------------------------
def unit_checks(failures):
    print("\n-- unit checks -------------------------------------------")
    from palm_postproc.steps.splittime import _parse_duration
    from palm_postproc.steps.splitvar import _detect_mode
    from palm_postproc.steps.coord import _parse_cf_epoch, is_temperature
    from palm_postproc.config import _parse_domain_suffix
    import numpy as np

    check(_parse_duration("1h") == 3600.0, "_parse_duration('1h')", failures)
    check(_parse_duration("30m") == 1800.0, "_parse_duration('30m')", failures)
    check(_parse_duration("3600") == 3600.0, "_parse_duration bare seconds",
          failures)

    check(_detect_mode(f"{CASE}_av_xy_N02") == "2D", "_detect_mode 2D",
          failures)
    check(_detect_mode(f"{CASE}_av_3d_N02") == "3D", "_detect_mode 3D",
          failures)
    check(_detect_mode(f"{CASE}_pr") is None, "_detect_mode other -> None",
          failures)

    for val, want in ((2, "_N02"), ("2", "_N02"), ("02", "_N02"),
                      ("N02", "_N02"), ("_N02", "_N02"), (None, ""),
                      (0, ""), ("0", "")):
        check(_parse_domain_suffix(val) == want,
              f"_parse_domain_suffix({val!r}) -> {want!r}", failures)

    # UTC offsets must be APPLIED, not stripped, or the numeric and decoded
    # paths disagree by the offset.
    e_utc = _parse_cf_epoch("seconds since 2023-08-23 09:00:00")
    e_off = _parse_cf_epoch("seconds since 2023-08-23 09:00:00+02:00")
    check(e_utc is not None and e_off is not None
          and (e_utc - e_off) / np.timedelta64(1, "s") == 7200.0,
          "_parse_cf_epoch applies a UTC offset", failures)

    # Same prefix list as palm2gis -- this is the divergence that used to
    # leave ta_2m* and t_surf in kelvin next to degrees-C neighbours.
    for name in ("theta", "theta_2m*_xy", "tsurf_xy", "t_surf*_xy",
                 "ta_2m*_xy", "ta"):
        check(is_temperature(name), f"is_temperature({name!r})", failures)
    for name in ("wspeed_xy", "u", "ind_z_xy"):
        check(not is_temperature(name), f"not is_temperature({name!r})",
              failures)

    from palm_postproc.steps.join import _merge_tolerance

    check(_merge_tolerance([0.0, 3600.0, 7200.0], 1800, None) == 900.0,
          "_merge_tolerance honours output_timestep", failures)
    check(_merge_tolerance([0.0, 3600.0, 7200.0], 0, None) == 1.0,
          "_merge_tolerance auto is capped at 1 s", failures)
    check(_merge_tolerance([0.0, 2.0, 4.0], 0, None) == 0.5,
          "_merge_tolerance auto = 1/4 median below the cap", failures)
    # An irregular axis must not merge two genuine records 100 s apart.
    check(_merge_tolerance([0.0, 100.0, 3600.0, 7200.0], 0, None) < 100.0,
          "_merge_tolerance does not over-merge an irregular axis", failures)
    check(_merge_tolerance([0.0, 3600.0], 1800, 5.0) == 5.0,
          "_merge_tolerance explicit override wins", failures)
    check(_merge_tolerance([42.0], 0, None) > 0.0,
          "_merge_tolerance survives a single timestep", failures)


# ------------------------------
# 5b. JOIN REGRESSION: INTERLEAVED RESTART CYCLES
# ------------------------------
def join_interleave_check(root, failures):
    """
    A re-run restart cycle whose output times fall *between* the times of
    the cycle it overlaps used to raise
    IndexError: size of data array does not conform to slice
    because each part was written with one contiguous slice assignment.
    """
    print("\n-- join: interleaved restart cycles -----------------------")
    import logging
    import numpy as np
    from netCDF4 import Dataset
    from palm_postproc.steps.join import _join_file

    src = root / "interleave" / "OUTPUT"
    dst = root / "interleave" / "OUTPUT_join"
    src.mkdir(parents=True, exist_ok=True)

    parts = ([1800.0, 3600.0], [5400.0, 7200.0, 9000.0], [6300.0, 8100.0])
    for i, times in enumerate(parts):
        ds = Dataset(src / f"{CASE}_av_3d.{i:03d}.nc", "w", format="NETCDF4")
        ds.createDimension("time", None)
        ds.createDimension("x", 4)
        ds.createDimension("y", 3)
        ds.createDimension("zu_3d", 2)
        ds.createVariable("time", "f8", ("time",))[:] = np.array(times)
        for name, size in (("x", 4), ("y", 3), ("zu_3d", 2)):
            ds.createVariable(name, "f4", (name,))[:] = np.arange(size)
        v = ds.createVariable("theta", "f4", ("time", "zu_3d", "y", "x"),
                              fill_value=-9999.0)
        v[:] = np.full((len(times), 2, 3, 4), float(i), dtype="f4")
        ds.close()

    log = logging.getLogger("palm_postproc")
    try:
        ok = _join_file(1, 1, f"{CASE}_av_3d", str(src), str(dst),
                        "", ".nc", "filenum", True, 0, 0, 0, None, 4,
                        True, False, log)
    except Exception as exc:
        check(False, f"join raised {type(exc).__name__}: {exc}", failures)
        return

    check(ok, "join of interleaved cycles returned OK", failures)

    with Dataset(dst / f"{CASE}_av_3d.nc") as ds:
        times = [float(t) for t in ds.variables["time"][:]]
        theta = ds.variables["theta"][:]

    check(times == [1800.0, 3600.0, 5400.0, 6300.0, 7200.0, 8100.0, 9000.0],
          f"join built the full timestep union, got {times}", failures)
    check(not np.ma.is_masked(theta) or theta.mask.sum() == 0,
          "no unwritten timesteps left in the joined data", failures)
    check([float(theta[i, 0, 0, 0]) for i in range(len(times))]
          == [0.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0],
          "each timestep came from the part that owns it", failures)


# ------------------------------
# 6. OUTPUT VERIFICATION
# ------------------------------
def verify_chain(base, failures):
    print("\n-- chain mode outputs -----------------------------------")
    import numpy as np
    import xarray as xr

    join_dir = base / CASE / "OUTPUT_join"
    coord_dir = base / CASE / "OUTPUT_coord"

    joined = sorted(join_dir.glob("*.nc"))
    check(len(joined) == 2, f"join wrote 2 file(s), got {len(joined)}",
          failures)

    if joined:
        with xr.open_dataset(join_dir / f"{CASE}_av_xy_N02.nc",
                             decode_times=False) as ds:
            # union of the two parts' times, duplicate 1200 s collapsed
            check(ds.sizes["time"] == 5,
                  f"join merged timesteps to 5, got {ds.sizes['time']}",
                  failures)

    produced = sorted(p.name for p in coord_dir.glob("*.nc"))
    check(len(produced) > 0, f"coord wrote {len(produced)} file(s)", failures)

    # splitvar: '*' stripped from the filename, uvw grouped
    names = " ".join(produced)
    check("*" not in names, "no '*' in output filenames", failures)
    check(any(".uvw." in n for n in produced),
          "u/v/w grouped into a .uvw file", failures)
    check(any("theta_2m_xy" in n for n in produced),
          "2D variable split out by name", failures)

    uvw = [p for p in coord_dir.glob("*.uvw.*.nc")]
    if check(len(uvw) == 1, "exactly one .uvw output", failures):
        with xr.open_dataset(uvw[0], decode_times=False) as ds:
            # splitz on BOTH vertical coords: z_max 45 keeps zu_3d at
            # 5,15,25,35,45 -> 5 levels and zw_3d at 0,10,20,30,40 -> 5
            check(ds.sizes.get("zu_3d") == 5,
                  f"splitz zu_3d -> 5 levels, got {ds.sizes.get('zu_3d')}",
                  failures)
            check(ds.sizes.get("zw_3d") == 5,
                  f"splitz ALSO sliced staggered zw_3d -> 5, got "
                  f"{ds.sizes.get('zw_3d')}", failures)

    xy = [p for p in coord_dir.glob("*_av_xy_N02.theta_2m_xy.utm.nc")]
    if check(len(xy) == 1, "2D theta output present", failures):
        with xr.open_dataset(xy[0], decode_times=False) as ds:
            # coord: UTM offsets applied once
            check(abs(float(ds.x[0]) - (ORIGIN_X + 5.0)) < 1e-6,
                  "coord applied origin_x to x", failures)
            check(abs(float(ds.y[0]) - (ORIGIN_Y + 5.0)) < 1e-6,
                  "coord applied origin_y to y", failures)
            # WGS84 auxiliary coordinates (crs: both)
            check("latitude" in ds and "longitude" in ds,
                  "crs: both wrote 2-D latitude/longitude", failures)
            if "latitude" in ds:
                check(ds["latitude"].dims == ("y", "x"),
                      "latitude is a 2-D aux coord on (y, x)", failures)
                check(45.0 < float(ds["latitude"].mean()) < 55.0,
                      "latitude lands in central Europe", failures)
            var = ds["theta_2m*_xy"]   # '*' is stripped from the
            #                            filename only, not the variable name
            # POTENTIAL temperature is exempt from `celsius`: it is a
            # kelvin-defined quantity, and "potential temperature in
            # degrees C" is a confusing thing to hand to anyone. Actual
            # temperatures (ta*, t_surf*, tsurf*) are checked below.
            check(var.attrs.get("units") == "K",
                  "theta_2m stays in kelvin (exempt from celsius)",
                  failures)
            vals = np.asarray(var.values)
            check(abs(float(np.nanmax(vals)) - 295.0) < 1e-3,
                  "theta values are not shifted by 273.15", failures)
            # the NoData corner must survive untouched
            check(abs(float(vals[0, 0, 0]) - FILL) < 1e-3
                  or not np.isfinite(vals[0, 0, 0]),
                  "fill value NOT shifted by 273.15", failures)
            # splittime: 20 min out of 10 min records -> every other step
            check(ds.sizes["time"] == 3,
                  f"splittime kept 3 of 5 records, got {ds.sizes['time']}",
                  failures)
            # time axis referenced to origin_time, epoch shift applied
            check(abs(float(ds.time[0])) < 1e-6,
                  f"first record at t=0 s after origin_time, got "
                  f"{float(ds.time[0])}", failures)
            # provenance
            check("palm_postproc_version" in ds.attrs,
                  "provenance: version attribute written", failures)
            check("palm_postproc_config_sha" in ds.attrs,
                  "provenance: config hash written", failures)
            check("history" in ds.attrs, "provenance: history written",
                  failures)

    # ta_* and t_surf* must be converted too (the old prefix list missed
    # them, so a directory mixed degrees C and kelvin)
    for pat, label in ((f"*.ta_2m_xy.utm.nc", "ta_2m"),
                       (f"*.t_surf_xy.utm.nc", "t_surf")):
        hits = list(coord_dir.glob(pat))
        if check(len(hits) == 1, f"{label} output present", failures):
            with xr.open_dataset(hits[0], decode_times=False) as ds:
                v = [n for n in ds.data_vars if n != "crs"][0]
                check(ds[v].attrs.get("units") == "degrees_C",
                      f"{label} converted to degrees_C too", failures)

    # time-invariant variable keeps no unlimited time dimension
    inv = list(coord_dir.glob("*ind_z_xy*.nc"))
    if inv:
        with xr.open_dataset(inv[0], decode_times=False) as ds:
            check("time" not in ds.dims,
                  "time-invariant variable has no time dimension", failures)


def verify_resume(base, cfg_path, failures):
    print("\n-- resume ------------------------------------------------")
    from palm_postproc.config import load as load_config
    from palm_postproc.pipeline import run as run_pipeline

    coord_dir = base / CASE / "OUTPUT_coord"
    before = {p: p.stat().st_mtime_ns for p in coord_dir.glob("*.nc")}

    cfg = load_config(cfg_path)
    run_pipeline(cfg, dry_run=False, only=None)

    after = {p: p.stat().st_mtime_ns for p in coord_dir.glob("*.nc")}
    check(before == after, "second run rewrote nothing (resume works)",
          failures)


def verify_classic(root, failures):
    """chain: false -- the path that used to die on cfg.paths.input."""
    print("\n-- classic (chain: false) mode ---------------------------")
    from palm_postproc.config import load as load_config
    from palm_postproc.pipeline import run as run_pipeline

    base = Path(root) / "classic"
    base.mkdir(parents=True, exist_ok=True)
    build_dataset(base)
    cfg_path = make_config(base, base / "classic.yaml", extra="")
    text = cfg_path.read_text().replace("chain: true", "chain: false")
    cfg_path.write_text(text)

    try:
        cfg = load_config(cfg_path)
        run_pipeline(cfg, dry_run=False, only=None)
        ok = True
    except AttributeError as exc:
        print(f"  FAIL    classic mode raised AttributeError: {exc}")
        failures.append("classic mode AttributeError")
        ok = False
    except Exception as exc:
        print(f"  FAIL    classic mode raised {type(exc).__name__}: {exc}")
        failures.append(f"classic mode {type(exc).__name__}")
        ok = False

    if ok:
        check(True, "classic mode ran without raising", failures)
        for step in ("splitvar", "splitz", "splittime", "coord"):
            d = base / CASE / f"OUTPUT_{step}"
            check(d.is_dir() and any(d.glob("*.nc")),
                  f"classic mode produced OUTPUT_{step}/", failures)


def verify_only(root, failures):
    """--only splitvar -- the other path through the classic runner."""
    print("\n-- --only splitvar ---------------------------------------")
    from palm_postproc.config import load as load_config
    from palm_postproc.pipeline import run as run_pipeline

    base = Path(root) / "only"
    base.mkdir(parents=True, exist_ok=True)
    build_dataset(base)
    cfg_path = make_config(base, base / "only.yaml")

    cfg = load_config(cfg_path)
    try:
        # join first so splitvar has something to read
        run_pipeline(cfg, dry_run=False, only={"join"})
        run_pipeline(cfg, dry_run=False, only={"splitvar"})
        check(True, "--only splitvar ran without raising", failures)
        d = base / CASE / "OUTPUT_splitvar"
        check(d.is_dir() and any(d.glob("*.nc")),
              "--only splitvar produced output", failures)
    except Exception as exc:
        print(f"  FAIL    --only splitvar raised {type(exc).__name__}: {exc}")
        failures.append(f"--only splitvar {type(exc).__name__}")


def verify_dry_run(root, failures):
    print("\n-- dry run -----------------------------------------------")
    from palm_postproc.config import load as load_config
    from palm_postproc.pipeline import run as run_pipeline

    base = Path(root) / "dry"
    base.mkdir(parents=True, exist_ok=True)
    build_dataset(base)
    cfg_path = make_config(base, base / "dry.yaml")

    cfg = load_config(cfg_path)
    run_pipeline(cfg, dry_run=True, only=None)
    coord_dir = base / CASE / "OUTPUT_coord"
    check(not coord_dir.exists() or not any(coord_dir.glob("*.nc")),
          "dry run wrote no output files", failures)


# ------------------------------
# 7. MAIN
# ------------------------------
def run(keep=False):
    if not check_dependencies():
        return 1

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from palm_postproc import __version__
    from palm_postproc.log import setup_logging
    setup_logging(verbosity="warning")
    print(f"\npalm_postproc {__version__}")

    failures = []
    tmp = Path(tempfile.mkdtemp(prefix="palm_postproc_test_"))
    try:
        unit_checks(failures)
        join_interleave_check(tmp, failures)

        print("\n-- building the synthetic dataset ------------------------")
        base = tmp / "main"
        base.mkdir(parents=True)
        out = build_dataset(base)
        print(f"  ok      {len(list(out.glob('*.nc')))} part file(s) in "
              f"{out}")

        print("\n-- running the pipeline (chain mode) ---------------------")
        from palm_postproc.config import load as load_config
        from palm_postproc.pipeline import run as run_pipeline
        cfg_path = make_config(base, base / "selftest.yaml")
        cfg = load_config(cfg_path)
        run_pipeline(cfg, dry_run=False, only=None)
        print("  ok      pipeline finished")

        verify_chain(base, failures)
        verify_resume(base, cfg_path, failures)
        verify_classic(tmp, failures)
        verify_only(tmp, failures)
        verify_dry_run(tmp, failures)

    except Exception:
        print("\nUNEXPECTED ERROR:\n")
        traceback.print_exc()
        failures.append("unexpected exception")
    finally:
        if keep:
            print(f"\nTemporary tree kept at: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 58)
    if failures:
        print(f"FAILED: {len(failures)} check(s) did not pass:")
        for f in failures:
            print(f"  - {f}")
        print("=" * 58)
        return 1
    print("All checks passed.")
    print("=" * 58)
    return 0


def main():
    ap = argparse.ArgumentParser(description="palm_postproc self-test")
    ap.add_argument("--keep", action="store_true",
                    help="keep the temporary tree and print its path")
    args = ap.parse_args()
    return run(keep=args.keep)


if __name__ == "__main__":
    sys.exit(main())
