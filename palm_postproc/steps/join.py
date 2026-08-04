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
    log.debug("[join]   Parts: %s", [Path(p).name for p in parts])
    return parts


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
            log.debug("[join]   Dimension: %-20s size=%s", d.name, d.size)
            if dn == "time":
                nc_out.createDimension(d.name, None)
            else:
                nc_out.createDimension(d.name, d.size)

        for vn in nc_in.variables:
            log.debug("[join]   Variable:  %s", vn)
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
                    log.warning("[join]   Too many dimensions in variable %s", vn)
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
        return True
    except Exception as exc:
        log.error("[join] Error copying NC structure: %s", exc)
        return False


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
    complevel:      int,
    overwrite:      bool,
    dry_run:        bool,
    log:            logging.Logger,
) -> bool:
    """
    Join all parts of one PALM output file into a single NetCDF.
    Returns True on success. Logic unchanged from palm_joinoutputs.py.
    """
    parts = _partnames(origpath, origfile, convention, file_suffix, file_prefix, log)
    parts = [p for p in parts if os.path.isfile(p)]

    prefix = f"[join] [{i:>{len(str(n))}}/{n}]"

    if not parts:
        log.warning("%s No part files found for %s — skipping.", prefix, origfile)
        return True

    fout     = os.path.join(finalpath, file_prefix + origfile + file_suffix)
    out_path = Path(fout)

    if not should_write(out_path, overwrite, log):
        log.info("%s %-45s skipped (up to date)", prefix, out_path.name)
        return True

    if dry_run:
        log.info("%s %-45s [DRY RUN] %d part(s) would be joined",
                 prefix, out_path.name, len(parts))
        return True

    log.info("%s %s", prefix, out_path.name)
    t0 = time.monotonic()

    Path(finalpath).mkdir(parents=True, exist_ok=True)

    # --- Phase 1: copy structure -------------------------------------------
    log.debug("[join]   Phase 1/3: copying structure from %s", Path(parts[0]).name)
    if create_new:
        nc_out = Dataset(fout, "w", format="NETCDF4")
        nc_in  = Dataset(parts[0], "r", format="NETCDF4")
        if not _nc_copy_structure(nc_in, nc_out, complevel, log):
            log.error("[join]   Failed to copy NC structure — aborting %s", out_path.name)
            nc_in.close()
            nc_out.close()
            return False
        nc_in.close()
    else:
        nc_out = Dataset(fout, "a", format="NETCDF4")

    # Identify time-dependent variables
    nc_vars = nc_out.variables
    vcp = [
        v for v in nc_vars
        if (len(nc_vars[v].dimensions) > 1
            or (len(nc_vars[v].dimensions) == 1 and nc_vars[v].dimensions[0] == "time"))
        and nc_vars[v].dimensions[0] == "time"
    ]
    log.debug("[join]   Time-dependent variables (%d): %s", len(vcp), vcp)

    # Some PALM output types (e.g. _topo_surf) are static and carry no
    # 'time' variable at all. Phase 1 already copied everything needed
    # from parts[0]; there is nothing to stitch across parts, so finish here.
    if "time" not in nc_vars:
        log.debug("[join]   No 'time' variable in %s — static file, structure copy is complete.",
                  Path(parts[0]).name)
        nc_out.close()
        log.info("%s   ✓  %s  (%s)  [static, no time dimension]",
                 prefix, fmt_size(out_path), fmt_elapsed(t0))
        return True

    # --- Phase 2: analyse timesteps ----------------------------------------
    log.debug("[join]   Phase 2/3: analysing timesteps in %d part(s)", len(parts))
    pinfo: dict = {}
    for ip, part in enumerate(parts):
        ptshift = ip * part_timeshift
        log.debug("[join]   Part %d/%d: %s  shift=%d", ip + 1, len(parts), Path(part).name, ptshift)
        ncp = Dataset(part, "r", format="NETCDF4")

        if "time" not in ncp.variables:
            log.warning("[join]   Part %s has no 'time' variable but %s does — skipping this part.",
                        Path(part).name, Path(parts[0]).name)
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
                    log.warning("[join]   Discontinuous timestep %d (t=%s) in %s",
                                i2, ptime[i2], Path(part).name)
        if ptsteps:
            pinfo[part] = {
                "nts":     len(ptsteps),
                "file":    part,
                "ptsteps": ptsteps,
                "offset":  offset,
                "ptshift": ptshift,
            }
            log.debug("[join]   → %d valid timesteps  (t=%.0f … %.0f)",
                      len(ptsteps), ptsteps[0], ptsteps[-1])
        ncp.close()

    if not pinfo:
        log.warning("[join]   No part contributed valid timesteps for %s — "
                    "output file will have an empty time dimension.", out_path.name)
        nc_out.close()
        return True

    # Assemble global timestep list
    tsteps_set: set = set()
    for part in pinfo:
        tsteps_set = tsteps_set.union(set(pinfo[part]["ptsteps"]))
    tsteps2 = sorted(tsteps_set)

    # Remove duplicate timesteps
    tsteps: list = []
    for i2, ts in enumerate(tsteps2):
        if i2 == 0:
            tsteps = [ts]
        elif abs(ts - tsteps2[i2 - 1]) < output_timestep / 2.0:
            tsteps[-1] = ts
        else:
            tsteps.append(ts)

    if output_timestep > 1:
        tsteps = [
            ts for ts in tsteps
            if int(ts) - int(ts / output_timestep) * output_timestep < 1
        ]

    log.debug("[join]   Global timesteps: %d  (t=%.0f … %.0f)",
              len(tsteps), tsteps[0] if tsteps else 0, tsteps[-1] if tsteps else 0)

    # Match parts to global timesteps
    for part in pinfo:
        ptsteps = pinfo[part]["ptsteps"]
        for tstep in ptsteps:
            if tstep in tsteps:
                break
        pinfo[part]["ptsmin"] = ptsteps.index(tstep) + pinfo[part]["offset"]
        pinfo[part]["tsmin"]  = tsteps.index(tstep)
        for tstep in ptsteps[ptsteps.index(tstep):]:
            if tstep not in tsteps:
                break
        if tstep not in tsteps:
            tstep = ptsteps[ptsteps.index(tstep) - 1]
        pinfo[part]["ptsmax"] = ptsteps.index(tstep) + pinfo[part]["offset"]
        pinfo[part]["tsmax"]  = tsteps.index(tstep)

    # --- Phase 3: copy data ------------------------------------------------
    log.debug("[join]   Phase 3/3: copying data from %d part(s)", len(pinfo))
    for ip, part in enumerate(pinfo):
        pi = pinfo[part]
        n_ts = pi["ptsmax"] - pi["ptsmin"] + 1
        log.debug("[join]   Part %d/%d: %s  ts[%d:%d] → global[%d:%d]  (%d timesteps)",
                  ip + 1, len(pinfo), Path(pi["file"]).name,
                  pi["ptsmin"], pi["ptsmax"], pi["tsmin"], pi["tsmax"], n_ts)
        ncp   = Dataset(pi["file"], "r", format="NETCDF4")
        pvars = ncp.variables

        times = pvars["time"][pi["ptsmin"]:pi["ptsmax"] + 1] + pi["ptshift"]
        nc_vars["time"][pi["tsmin"]:pi["tsmax"] + 1] = times

        for v in vcp:
            if v not in pvars:
                log.warning("[join]   Variable %s not in part %s — skipping.",
                            v, Path(part).name)
                continue
            nd     = len(nc_vars[v].dimensions)
            vs     = nc_vars[v].shape
            offset = pi["ptsmin"] - pi["tsmin"]
            for i2 in range(pi["tsmin"], pi["tsmax"] + 1):
                if nd == 1:
                    nc_vars[v][i2] = pvars[v][i2 + offset]
                elif nd == 2:
                    nc_vars[v][i2, :] = pvars[v][i2 + offset, 0:vs[1]]
                elif nd == 3:
                    nc_vars[v][i2, :, :] = pvars[v][i2 + offset, 0:vs[1], 0:vs[2]]
                elif nd == 4:
                    nc_vars[v][i2, :, :, :] = pvars[v][i2 + offset, 0:vs[1], 0:vs[2], 0:vs[3]]
                elif nd == 5:
                    nc_vars[v][i2, :, :, :, :] = pvars[v][i2 + offset, 0:vs[1], 0:vs[2], 0:vs[3], 0:vs[4]]
                elif nd == 6:
                    nc_vars[v][i2, :, :, :, :, :] = pvars[v][i2 + offset, 0:vs[1], 0:vs[2], 0:vs[3], 0:vs[4], 0:vs[5]]
                else:
                    log.warning("[join]   Too many dimensions in variable %s — skipping.", v)
        ncp.close()

    nc_out.close()
    log.info("%s   ✓  %s  (%s)", prefix, fmt_size(out_path), fmt_elapsed(t0))
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
    log.debug("[join] Auto-detected filelist: %s", result)
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
    log.info("[join] convention: %s", step_cfg.convention)
    log.info("[join] suffix: %s", step_cfg.file_suffix or "(none)")
    log.info("[join] complevel: %d", step_cfg.complevel)

    if not Path(origpath).is_dir():
        raise RuntimeError(f"[join] Input directory does not exist: {origpath}")

    # Resolve filelist
    filelist = step_cfg.filelist
    if filelist == "all" or filelist is None:
        filelist = _detect_filelist(origpath, step_cfg.file_suffix, log)
        if not filelist:
            log.warning("[join] No files auto-detected in %s", origpath)
            return
        log.info("[join] Auto-detected %d file(s) to join:", len(filelist))
    else:
        filelist = [f.replace("{case}", cfg.case) for f in filelist]
        log.info("[join] %d file(s) to join (from config):", len(filelist))

    for f in filelist:
        log.info("[join]   %s", f)

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
            complevel       = step_cfg.complevel,
            overwrite       = cfg.overwrite or force,
            dry_run         = dry_run,
            log             = log,
        )
        if not ok:
            failures += 1

    n_ok = n - failures
    log.info("[join] Done.  %d/%d file(s) joined successfully  (%s)",
             n_ok, n, fmt_elapsed(t_step))

    if failures:
        raise RuntimeError(f"[join] {failures} file(s) failed — check log above.")
