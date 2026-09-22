"""
palm_postproc.steps.join
-------------------------
Joins segmented PALM NetCDF output files into single files.
Refactored from palm_joinoutputs.py — logic unchanged.

Runs before all other steps. Its output directory becomes the input
for splitvar → splitz → splittime → coord.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import time
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from ..config import Config
from ..log import step
from ..utils import fmt_elapsed, fmt_size, should_write


# ---------------------------------------------------------------------------
# Helpers (logic unchanged from palm_joinoutputs.py)
# ---------------------------------------------------------------------------

def _partnames(
    origpath:    str,
    origfile:    str,
    convention:  str,
    file_suffix: str,
    file_prefix: str,
    log:         logging.Logger,
) -> list[str]:
    parts = []
    if convention == "dirpart":
        parts.extend(glob.glob(os.path.join(origpath + ".part*", origfile + file_suffix)))
        parts.sort()
        parts.insert(0, os.path.join(origpath, origfile + file_suffix))
    elif convention == "filepart":
        parts.extend(glob.glob(os.path.join(origpath, origfile + ".part*" + file_suffix)))
        parts.sort()
        parts.insert(0, os.path.join(origpath, origfile + file_suffix))
    elif convention == "tmpdir":
        parts.extend(glob.glob(os.path.join(origpath + ".*", origfile + file_suffix)))
        parts.sort()
    elif convention == "filenum":
        parts.extend(glob.glob(os.path.join(origpath, origfile + ".*" + file_suffix)))
        parts.sort()
    elif convention == "filepattern":
        parts.extend(glob.glob(os.path.join(origpath, file_prefix + origfile + file_suffix + "*")))
        parts.sort()
    elif convention == "singlefile":
        parts = [os.path.join(origpath, origfile + file_suffix)]
    log.debug("[join] parts: %s", [Path(p).name for p in parts])
    return parts


# Upper bound (seconds) on the automatically derived timestep merge
# tolerance. Restart-boundary jitter is sub-second; anything wider is a
# real timestep and must not be merged without an explicit merge_tol.
AUTO_MERGE_TOL_MAX = 1.0


def _merge_tolerance(tsteps: list, output_timestep: int,
                     merge_tol: float | None) -> float:
    """
    Tolerance (seconds) within which two timesteps count as the same instant.

    Explicit config wins. Otherwise output_timestep/2 if set, else a quarter
    of the median output interval, capped at AUTO_MERGE_TOL_MAX.

    The cap matters. The jitter this is meant to absorb is the sub-second
    difference PALM writes at restart-cycle boundaries, so a tolerance of
    order one second is all that is ever needed. A quarter of the median
    interval alone is far too generous on an irregular time axis — on
    t = [0, 100, 3600, 7200] it comes out at 887 s and would silently merge
    the 0 s and 100 s records into one. Under-merging only leaves a
    cosmetic near-duplicate record; over-merging destroys data, so this
    errs low and leaves merge_tol for anything wider.
    """
    if merge_tol is not None:
        return float(merge_tol)
    if output_timestep > 0:
        return output_timestep / 2.0
    arr = np.asarray(tsteps, dtype=float)
    if arr.size < 2:
        return 1e-6
    diffs = np.diff(arr)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 1e-6
    return max(min(float(np.median(diffs)) * 0.25, AUTO_MERGE_TOL_MAX), 1e-6)


def _shape_ok(vn, dst_var, src_var, partname: str, log: logging.Logger) -> bool:
    """Check that a part's variable is large enough to fill the output slot."""
    dshape = dst_var.shape[1:]
    sshape = src_var.shape[1:]
    if len(dshape) != len(sshape) or any(s < d for s, d in zip(sshape, dshape)):
        log.warning("[join] %s in %s has shape %s, output needs %s - "
                    "skipped for this part.", vn, partname, sshape, dshape)
        return False
    return True


# Variables larger than this are not compared across parts. Nothing PALM
# writes as a time-invariant coordinate comes close (the largest is the
# surface element list, order 1e5), so the cap only ever skips the bulk
# static fields of a _topo_surf file, which is single-part anyway.
GEOM_CHECK_MAX_SIZE = 5_000_000


def _arrays_equal(a, b) -> bool:
    """Value-and-mask comparison that tolerates NaN and masked entries."""
    ma, mb = np.ma.getmaskarray(a), np.ma.getmaskarray(b)
    if ma.shape != mb.shape or not np.array_equal(ma, mb):
        return False
    da, db = np.ma.getdata(a), np.ma.getdata(b)
    if da.shape != db.shape:
        return False
    if da.dtype.kind == "f" and db.dtype.kind == "f":
        return np.array_equal(da, db, equal_nan=True)
    return np.array_equal(da, db)


def _check_part_geometry(parts: list[str], log: logging.Logger) -> str | None:
    """Verify every part describes the SAME grid / surface elements.

    Phase 1 copies all time-invariant variables from parts[0] alone and
    never revisits them; Phase 3 then appends every part's data along
    time. That is correct for restart-cycle parts, which differ only in
    the times they cover. It is silently wrong for parts that hold a
    different *piece of space* — a spatially decomposed surface output,
    say — because parts[0]'s xs/ys/zs would be paired with another part's
    values. Nothing downstream can detect that: the file is structurally
    valid and every check passes. So it has to be caught here.

    Returns None when the parts agree, or a message naming the first
    variable that does not.
    """
    if len(parts) < 2:
        return None

    with Dataset(parts[0], "r", format="NETCDF4") as nc0:
        ref_names = [
            vn for vn, v in nc0.variables.items()
            if not v.dimensions or v.dimensions[0] != "time"
        ]
        ref = {}
        for vn in ref_names:
            v = nc0.variables[vn]
            if v.size > GEOM_CHECK_MAX_SIZE:
                log.debug("[join] %s has %d elements - too large to compare "
                          "across parts, not checked.", vn, v.size)
                continue
            ref[vn] = v[...]

    if not ref:
        return None

    log.debug("[join] comparing %d time-invariant variable(s) across %d "
              "part(s)", len(ref), len(parts))

    for part in parts[1:]:
        with Dataset(part, "r", format="NETCDF4") as ncp:
            for vn, ref_val in ref.items():
                if vn not in ncp.variables:
                    return (f"part {Path(part).name} is missing the time-invariant "
                            f"variable '{vn}' that {Path(parts[0]).name} defines")
                if not _arrays_equal(ref_val, ncp.variables[vn][...]):
                    return (f"time-invariant variable '{vn}' differs between "
                            f"{Path(parts[0]).name} and {Path(part).name}")
    return None


def _nc_copy_structure(
    nc_in:     Dataset,
    nc_out:    Dataset,
    complevel: int,
    log:       logging.Logger,
) -> bool:
    """Copy structure and time-invariant data from nc_in to nc_out."""
    try:
        for attr_in in nc_in.ncattrs():
            nc_out.setncattr(attr_in, nc_in.getncattr(attr_in))

        for dn in nc_in.dimensions:
            d = nc_in.dimensions[dn]
            if dn == "time":
                nc_out.createDimension(d.name, None)
            else:
                nc_out.createDimension(d.name, d.size)

        for vn in nc_in.variables:
            v = nc_in.variables[vn]
            if len(v.dimensions) == 0:
                nc_out.createVariable(vn, v.datatype, v.dimensions)
                nc_out[vn].setncatts(nc_in[vn].__dict__)
                nc_out[vn][:] = nc_in[vn][:]
            elif v.dimensions[0] != "time":
                nc_out.createVariable(vn, v.datatype, v.dimensions)
                nc_out[vn].setncatts(nc_in[vn].__dict__)
                if len(v.dimensions) == 1:
                    nc_out[vn][:] = nc_in[vn][:]
                elif len(v.dimensions) == 2:
                    nc_out[vn][:, :] = nc_in[vn][:, :]
                elif len(v.dimensions) == 3:
                    nc_out[vn][:, :, :] = nc_in[vn][:, :, :]
                else:
                    log.warning("[join] %s has too many dimensions - not copied.", vn)
            elif vn == "time":
                nc_out.createVariable(vn, v.datatype, v.dimensions)
                nc_out[vn].setncatts(nc_in[vn].__dict__)
            else:
                if v.datatype == np.float32:
                    nc_out.createVariable(
                        vn, v.datatype, v.dimensions,
                        zlib=True, complevel=complevel, fill_value=-9999.0,
                    )
                else:
                    nc_out.createVariable(
                        vn, v.datatype, v.dimensions,
                        zlib=True, complevel=complevel,
                    )
                nc_out[vn].setncatts(nc_in[vn].__dict__)
        log.debug("[join] structure: %d dimension(s), %d variable(s)",
                  len(nc_in.dimensions), len(nc_in.variables))
        return True
    except Exception as exc:
        log.error("[join] could not copy the NetCDF structure: %s", exc)
        return False


def _stamp_join(nc_out: Dataset, cfg, cfg_hash: str, parts: list[str]) -> None:
    """Attach provenance attributes to a joined file, CF style.

    `history` is appended to, never replaced, so PALM's own history
    survives. `palm_postproc_stage` marks this as join output: coord
    output must never reach palm2gis, and a positive marker is a better
    test than guessing from what a file does not contain.
    """
    from ..utils import provenance_attrs
    from .. import __version__

    attrs = provenance_attrs(cfg, cfg_hash)
    attrs["palm_postproc_stage"] = "join"
    attrs["palm_postproc_source"] = os.pathsep.join(
        Path(p).name for p in parts)
    attrs["palm_postproc_n_parts"] = len(parts)

    line = (f"{attrs['palm_postproc_processed']}: palm_postproc "
            f"{__version__}: {attrs['palm_postproc_command']}")
    existing = ""
    if "history" in nc_out.ncattrs():
        existing = str(nc_out.getncattr("history")).rstrip()

    for k, v in attrs.items():
        nc_out.setncattr(k, v)
    nc_out.setncattr("history", f"{existing}\n{line}".strip() if existing else line)


def _join_file(
    i:              int,
    n:              int,
    origfile:       str,
    origpath:       str,
    finalpath:      str,
    file_prefix:    str,
    file_suffix:    str,
    convention:     str,
    create_new:     bool,
    offset_min:     int,
    part_timeshift: int,
    output_timestep: int,
    merge_tol:      float | None,
    complevel:      int,
    overwrite:      bool,
    dry_run:        bool,
    log:            logging.Logger,
    # Optional so a caller that only wants the join mechanics (the test
    # harness) need not build a Config just to stamp provenance.
    cfg:            Config | None = None,
    cfg_hash:       str = "",
) -> bool:
    """
    Join all parts of one PALM output file into a single NetCDF.
    Returns True on success. Logic unchanged from palm_joinoutputs.py.
    """
    parts = _partnames(origpath, origfile, convention, file_suffix, file_prefix, log)
    parts = [p for p in parts if os.path.isfile(p)]

    if not parts:
        log.warning("[join] no part files for %s - skipped.", origfile)
        return True

    fout     = os.path.join(finalpath, file_prefix + origfile + file_suffix)
    out_path = Path(fout)

    if not should_write(out_path, overwrite, log, "join"):
        return True

    if dry_run:
        log.info("[join] would write %s: %d part(s) (dry run)",
                 out_path.name, len(parts))
        return True

    log.debug("[join] %d/%d: joining %d part(s) -> %s",
              i, n, len(parts), out_path.name)
    t0 = time.monotonic()

    # Fail before writing anything, not halfway through.
    mismatch = _check_part_geometry(parts, log)
    if mismatch:
        # Join stitches parts along TIME and takes every other variable from
        # the first part - only valid for restart cycles of the same grid.
        # Spatially decomposed parts would pair one part's geometry with
        # another part's data.
        log.error("[join] %s - join restart cycles of one grid, not "
                  "subdomains.", mismatch)
        return False

    Path(finalpath).mkdir(parents=True, exist_ok=True)

    # --- Phase 1: copy structure -------------------------------------------
    log.debug("[join] phase 1/3: copying structure from %s", Path(parts[0]).name)
    if create_new:
        nc_out = Dataset(fout, "w", format="NETCDF4")
        nc_in  = Dataset(parts[0], "r", format="NETCDF4")
        if not _nc_copy_structure(nc_in, nc_out, complevel, log):
            log.error("[join] structure copy failed - %s aborted", out_path.name)
            nc_in.close()
            nc_out.close()
            return False
        nc_in.close()
    else:
        nc_out = Dataset(fout, "a", format="NETCDF4")

    # Provenance, written here so every exit path below carries it.
    #
    # An OUTPUT_join directory used to be the one anonymous output of this
    # tool: the filename said which PALM file it came from and nothing
    # about which config or which code produced it. It is also the
    # directory palm2gis reads, and `palm_postproc_stage` lets palm2gis
    # confirm a file is join output rather than infer it from the absence
    # of coord's fingerprints.
    if cfg is not None:
        _stamp_join(nc_out, cfg, cfg_hash, parts)

    # Identify time-dependent variables
    nc_vars = nc_out.variables
    vcp = [
        v for v in nc_vars
        if (len(nc_vars[v].dimensions) > 1
            or (len(nc_vars[v].dimensions) == 1 and nc_vars[v].dimensions[0] == "time"))
        and nc_vars[v].dimensions[0] == "time"
    ]
    log.debug("[join] time-dependent variables (%d): %s", len(vcp), vcp)

    # Some PALM output types (e.g. _topo_surf) are static and carry no
    # 'time' variable at all. Phase 1 already copied everything needed
    # from parts[0]; there is nothing to stitch across parts, so finish here.
    if "time" not in nc_vars:
        nc_out.close()
        log.info("[join] wrote %s: static (no time), %s in %s",
                 out_path.name, fmt_size(out_path), fmt_elapsed(t0))
        return True

    # --- Phase 2: analyse timesteps ----------------------------------------
    log.debug("[join] phase 2/3: analysing timesteps in %d part(s)", len(parts))
    pinfo: dict = {}
    for ip, part in enumerate(parts):
        ptshift = ip * part_timeshift
        log.debug("[join] part %d/%d: %s, shift %d", ip + 1, len(parts),
                  Path(part).name, ptshift)
        ncp = Dataset(part, "r", format="NETCDF4")

        if "time" not in ncp.variables:
            log.warning("[join] part %s has no 'time' variable - skipped.",
                        Path(part).name)
            ncp.close()
            continue

        ptime = ncp.variables["time"]

        ptsteps: list = []
        offset = offset_min
        for i2 in range(len(ptime)):
            if not ptime[i2].mask:
                offset = max(offset_min, i2)
                break
        if offset is not None:
            for i2 in range(offset, len(ptime)):
                if ptime[i2].mask:
                    break
                ptsteps.append(ptime[i2].data.item(0))
            for i2 in range(offset + len(ptsteps), len(ptime)):
                if not ptime[i2].mask:
                    log.warning("[join] discontinuous timestep %d (t=%s) in %s",
                                i2, ptime[i2], Path(part).name)
        if ptsteps:
            pinfo[part] = {
                "nts":     len(ptsteps),
                "file":    part,
                "ptsteps": ptsteps,
                "offset":  offset,
                "ptshift": ptshift,
            }
            log.debug("[join] %d valid timestep(s), t = %.0f .. %.0f",
                      len(ptsteps), ptsteps[0], ptsteps[-1])
        ncp.close()

    if not pinfo:
        log.warning("[join] no valid timesteps for %s - its time axis is "
                    "empty.", out_path.name)
        nc_out.close()
        return True

    # Assemble global timestep list
    tsteps_set: set = set()
    for part in pinfo:
        tsteps_set = tsteps_set.union(set(pinfo[part]["ptsteps"]))
    tsteps2 = sorted(tsteps_set)

    # Collapse near-duplicate timesteps (restart-cycle overlaps).
    # The later value wins, matching the original behaviour.
    tol = _merge_tolerance(tsteps2, output_timestep, merge_tol)
    log.debug("[join] timestep merge tolerance: %g s", tol)

    tsteps: list = []
    for ts in tsteps2:
        if tsteps and abs(ts - tsteps[-1]) < tol:
            tsteps[-1] = ts
        else:
            tsteps.append(ts)

    if len(tsteps) < len(tsteps2):
        # Reported at info: merging collapses two records into one, which
        # changes the time axis palm2gis will read. On an hourly surface
        # file the auto tolerance is capped at 1 s, so anything merged
        # here is restart-boundary jitter; anything NOT merged that should
        # have been shows up downstream as a near-duplicate timestep.
        log.info("[join] merged %d near-duplicate timestep(s) within %g s",
                 len(tsteps2) - len(tsteps), tol)

    if output_timestep > 1:
        tsteps = [
            ts for ts in tsteps
            if int(ts) - int(ts / output_timestep) * output_timestep < 1
        ]

    if not tsteps:
        log.warning("[join] all timesteps of %s filtered out - nothing written.",
                    out_path.name)
        nc_out.close()
        return True

    log.debug("[join] %d global timestep(s), t = %.0f .. %.0f",
              len(tsteps), tsteps[0], tsteps[-1])

    # Map every part timestep to its global index.
    #
    # The previous implementation assumed each part covered a *contiguous*
    # run of the global timestep list and wrote it with a single slice
    # assignment. That assumption breaks whenever another part contributes
    # a timestep that falls between two of this part's timesteps — e.g. a
    # re-run restart cycle whose output times are offset from the cycle it
    # replaces. The slice was then longer than the data and netCDF4 raised
    # "size of data array does not conform to slice". An explicit
    # (source index → global index) map removes the assumption entirely.
    tarr = np.asarray(tsteps, dtype=float)
    for part in pinfo:
        pi      = pinfo[part]
        ptsteps = pi["ptsteps"]
        matched: dict = {}          # global index → source index (last wins)
        n_drop  = 0
        for j, ts in enumerate(ptsteps):
            k = int(np.argmin(np.abs(tarr - ts)))
            if abs(tarr[k] - ts) < tol:
                matched[k] = j + pi["offset"]
            else:
                n_drop += 1
        pi["tmap"] = sorted((src, dst) for dst, src in matched.items())
        if n_drop:
            log.debug("[join] %s: %d timestep(s) filtered out",
                      Path(part).name, n_drop)

    # --- Phase 3: copy data ------------------------------------------------
    log.debug("[join] phase 3/3: copying data from %d part(s)", len(pinfo))
    for ip, part in enumerate(pinfo):
        pi   = pinfo[part]
        tmap = pi["tmap"]
        if not tmap:
            log.warning("[join] part %s contributes no timesteps - skipped.",
                        Path(part).name)
            continue
        log.debug("[join] part %d/%d: %s -> %d timestep(s), global [%d..%d]",
                  ip + 1, len(pinfo), Path(pi["file"]).name,
                  len(tmap), tmap[0][1], tmap[-1][1])
        ncp   = Dataset(pi["file"], "r", format="NETCDF4")
        pvars = ncp.variables

        for src, dst in tmap:
            nc_vars["time"][dst] = float(pvars["time"][src]) + pi["ptshift"]

        for v in vcp:
            if v == "time":
                continue        # already written above, with ptshift applied
            if v not in pvars:
                log.warning("[join] %s not in part %s - skipped.",
                            v, Path(part).name)
                continue
            nd = len(nc_vars[v].dimensions)
            vs = nc_vars[v].shape
            if nd > 6:
                log.warning("[join] %s has too many dimensions - skipped.", v)
                continue
            if not _shape_ok(v, nc_vars[v], pvars[v], Path(part).name, log):
                continue
            sl = tuple(slice(0, vs[k]) for k in range(1, nd))
            for src, dst in tmap:
                nc_vars[v][(dst,) + sl] = pvars[v][(src,) + sl]
        ncp.close()

    nc_out.close()
    log.info("[join] wrote %s: %d part(s), %d time(s), %s in %s",
             out_path.name, len(pinfo), len(tsteps), fmt_size(out_path),
             fmt_elapsed(t0))
    return True


# ---------------------------------------------------------------------------
# Auto-detection of filelist
# ---------------------------------------------------------------------------

def _detect_filelist(origpath: str, file_suffix: str, log: logging.Logger) -> list[str]:
    """Scan origpath for PALM output files and return their stems."""
    candidates = set()
    pattern = os.path.join(origpath, "*" + file_suffix)
    for p in glob.glob(pattern):
        name = os.path.basename(p)
        stem = name[: -len(file_suffix)] if file_suffix else name
        stem = re.sub(r"\.\d{3}$", "", stem)
        candidates.add(stem)
    result = sorted(candidates)
    log.debug("[join] auto-detected filelist: %s", result)
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(cfg: Config, dry_run: bool, log: logging.Logger,
        force: bool = False) -> None:
    """
    Run the join step for all files in cfg.steps.join.filelist.
    Output lands in cfg.paths.output_join (= input for downstream steps).
    Raises RuntimeError if any file fails.

    force=True rewrites existing joined files even when cfg.overwrite is
    false. The pipeline sets it when the join settings changed since the
    last run, since the existing files were produced with different
    parameters and are therefore stale.
    """
    step(log, "Joining PALM output parts")

    step_cfg  = cfg.steps.join
    origpath  = str(cfg.paths.join_input)
    finalpath = str(cfg.paths.output_join)

    log.info("[join] input dir: %s", origpath)
    log.info("[join] output dir: %s", finalpath)
    log.debug("[join] convention %s, suffix %s, complevel %d",
              step_cfg.convention, step_cfg.file_suffix or "(none)",
              step_cfg.complevel)

    if not Path(origpath).is_dir():
        raise RuntimeError(f"[join] input directory does not exist: {origpath}")

    # Resolve filelist
    filelist = step_cfg.filelist
    if filelist == "all" or filelist is None:
        filelist = _detect_filelist(origpath, step_cfg.file_suffix, log)
        if not filelist:
            log.warning("[join] no files found in %s", origpath)
            return
        source = "auto-detected"
    else:
        filelist = [f.replace("{case}", cfg.case) for f in filelist]
        source = "from config"
    log.info("[join] %d file(s) to join (%s)", len(filelist), source)
    log.debug("[join] files: %s", ", ".join(filelist))

    # join_config_hash, not config_hash: join's outputs are the INPUT to
    # the post-join chain, so the digest that identifies them is the one
    # built from join's own settings.
    from ..state import join_config_hash
    cfg_h = join_config_hash(cfg)

    t_step = time.monotonic()
    failures = 0
    n = len(filelist)

    for idx, origfile in enumerate(filelist, 1):
        ok = _join_file(
            i               = idx,
            n               = n,
            origfile        = origfile,
            origpath        = origpath,
            finalpath       = finalpath,
            file_prefix     = step_cfg.file_prefix,
            file_suffix     = step_cfg.file_suffix,
            convention      = step_cfg.convention,
            create_new      = step_cfg.create_new_file,
            offset_min      = step_cfg.offset_min,
            part_timeshift  = step_cfg.part_timeshift,
            output_timestep = step_cfg.output_timestep,
            merge_tol       = step_cfg.merge_tol,
            complevel       = step_cfg.complevel,
            overwrite       = cfg.overwrite or force,
            dry_run         = dry_run,
            log             = log,
            cfg             = cfg,
            cfg_hash        = cfg_h,
        )
        if not ok:
            failures += 1

    # Counts written and kept files alike; failures raise below.
    if not failures:
        log.info("[join] %d file(s) done in %s", n, fmt_elapsed(t_step))

    if failures:
        raise RuntimeError(f"[join] {failures} file(s) failed - see the log above.")
