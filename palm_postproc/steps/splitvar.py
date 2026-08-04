"""
palm_postproc.steps.splitvar
----------------------------
Splits a PALM _av_xy or _av_3d file into one NetCDF per variable.
Refactored from palm_splitvar.py — logic unchanged.
"""

from __future__ import annotations

import re
import time
import logging
from pathlib import Path

import xarray as xr

from ..config import Config
from ..log import step
from ..utils import (
    open_dataset, write_dataset, try_dask_chunks,
    fmt_size, fmt_elapsed, should_write,
)

_CHUNKS_2D = {"time": 1}
_CHUNKS_3D = {"time": 1, "zu_3d": 10}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_mode(stem: str) -> str | None:
    """Return '2D', '3D', or None if the filename suffix is not recognised."""
    m = re.search(r"_(av_xy|av_3d)(_N\d+)?", stem)
    if not m:
        return None
    return "2D" if m.group(1) == "av_xy" else "3D"


def _resolve_vars(
    ds: xr.Dataset,
    requested: str | list[str] | None,
    log: logging.Logger,
    label: str,
) -> list[str]:
    """
    Resolve variable list from 'all', 'none', None (blank), or an explicit list.
    'none' / None / blank -> skip all variables of this type (return empty list).
    'all'                 -> return every variable in the dataset.
    [list]                -> return only variables that exist, warn about missing.
    """
    if requested is None or (isinstance(requested, str) and requested.strip().lower() == "none"):
        log.debug("[splitvar] vars_%s is none/blank — skipping %s files.", label.lower(), label)
        return []

    if requested == "all":
        return list(ds.data_vars)

    available = list(ds.data_vars)
    resolved  = []
    for name in requested:
        if name in available:
            resolved.append(name)
        else:
            log.warning("[splitvar] Variable not in dataset (%s), skipping: %s", label, name)
    return resolved


def _split_file(
    src_path:     Path,
    out_dir:      Path,
    cfg:          Config,
    dry_run:      bool,
    log:          logging.Logger,
) -> int:
    """
    Split one file.  Returns number of failures.
    """
    stem = src_path.stem
    mode = _detect_mode(stem)

    if mode is None:
        log.warning("[splitvar] Cannot detect mode from filename, skipping: %s", src_path.name)
        return 0

    log.debug("[splitvar] %s → mode=%s", src_path.name, mode)

    is_3d    = (mode == "3D")
    chunks   = try_dask_chunks(_CHUNKS_2D, _CHUNKS_3D, is_3d)
    step_cfg = cfg.steps.splitvar

    ds = open_dataset(src_path, chunks)

    requested = step_cfg.vars_3d if is_3d else step_cfg.vars_2d
    data_vars = _resolve_vars(ds, requested, log, label=mode)

    if not data_vars:
        log.warning("[splitvar] No valid variables found in %s — skipping.", src_path.name)
        return 0

    group_vars     = [v for v in step_cfg.group_vars if v in data_vars] if is_3d else []
    individual_vars = [v for v in data_vars if v not in group_vars]

    log.debug("[splitvar] group_vars=%s  individual=%s", group_vars, individual_vars)

    if dry_run:
        log.info("[splitvar] [DRY RUN] Would write %d file(s) from %s",
                 (1 if group_vars else 0) + len(individual_vars), src_path.name)
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0

    # ---- Combined vector file (3D only) -----------------------------------
    if group_vars:
        out_path = out_dir / f"{stem}.{step_cfg.group_suffix}.nc"
        if should_write(out_path, cfg.overwrite, log):
            log.info("[splitvar] Writing group %s → %s ...", group_vars, out_path.name)
            t0 = time.monotonic()
            try:
                write_dataset(ds[group_vars], out_path, cfg.complevel)
                log.info("[splitvar]   ✓  %s  (%s)", fmt_size(out_path), fmt_elapsed(t0))
            except Exception as exc:
                log.error("[splitvar] FAILED writing %s: %s", out_path.name, exc)
                failures += 1

    # ---- Individual scalar files ------------------------------------------
    for i, varname in enumerate(individual_vars, 1):
        safe_name = varname.replace("*", "")
        out_path  = out_dir / f"{stem}.{safe_name}.nc"
        if not should_write(out_path, cfg.overwrite, log):
            continue
        log.info("[splitvar] [%d/%d] %s → %s ...",
                 i, len(individual_vars), varname, out_path.name)
        t0 = time.monotonic()
        try:
            write_dataset(ds[[varname]], out_path, cfg.complevel)
            log.info("[splitvar]   ✓  %s  (%s)", fmt_size(out_path), fmt_elapsed(t0))
        except Exception as exc:
            log.error("[splitvar] FAILED writing %s: %s", out_path.name, exc)
            failures += 1

    ds.close()
    return failures


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(cfg: Config, dry_run: bool, log: logging.Logger) -> None:
    """
    Run the splitvar step for all .nc files in the pipeline's input directory.
    Raises RuntimeError if any file fails.

    The input is whatever feeds the post-join chain — the explicit
    ``paths.input`` override, the join output, or the raw PALM output when
    join is disabled. This used to read a non-existent ``cfg.paths.input``
    attribute, which made every classic-mode run (``chain: false``, or
    ``--only splitvar``) fail immediately with an AttributeError.
    """
    step(log, "Splitting files by variable")

    from ..pipeline import _chained_input_dir

    input_dir = _chained_input_dir(cfg)
    out_dir   = cfg.paths.output_splitvar

    nc_files = sorted(input_dir.glob("*.nc"))
    if not nc_files:
        log.warning("[splitvar] No .nc files found in %s", input_dir)
        return

    log.info("[splitvar] %d file(s) to process", len(nc_files))
    total_failures = 0

    for src_path in nc_files:
        log.info("[splitvar] Processing: %s", src_path.name)
        total_failures += _split_file(src_path, out_dir, cfg, dry_run, log)

    if total_failures:
        raise RuntimeError(f"[splitvar] {total_failures} file(s) failed — check log above.")

    log.info("[splitvar] Done.")
