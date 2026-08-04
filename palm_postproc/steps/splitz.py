"""
palm_postproc.steps.splitz
--------------------------
Restricts 3D PALM output files to z <= z_max.
Refactored from palm_splitz.py — logic unchanged.
"""

from __future__ import annotations

import time
import logging
from pathlib import Path

import xarray as xr

from ..config import Config
from ..log import step
from ..utils import (
    open_dataset, write_dataset, try_dask_chunks,
    fmt_size, fmt_elapsed, fmt_duration, should_write,
)

_CHUNKS_2D = {"time": 1}
_CHUNKS_3D = {"time": 1, "zu_3d": 10}

# Every vertical coordinate PALM writes into 3D output. A file may carry
# more than one of them at the same time: splitvar groups u, v and w into a
# single `.uvw.nc`, where u/v live on zu_3d and w on the STAGGERED zw_3d.
# Slicing only the configured coordinate left w at its full height in that
# file, so the group file mixed a 300 m scalar field with a full-column w.
_Z_COORDS = ("zu_3d", "zw_3d", "zu", "zw", "z")


# ---------------------------------------------------------------------------
# Core slice logic
# ---------------------------------------------------------------------------

def z_coords_present(ds: xr.Dataset, primary: str) -> list:
    """Vertical coordinates of *ds* that should be sliced.

    The configured `z_coord` comes first (it is the one reported in the log
    and the one whose absence makes the step a no-op), followed by any other
    known vertical coordinate the file carries.
    """
    found = [primary] if primary in ds.coords else []
    found += [c for c in _Z_COORDS if c in ds.coords and c not in found]
    return found


def _slice_by_z(ds: xr.Dataset, z_max: float, z_coord: str) -> xr.Dataset:
    coords = z_coords_present(ds, z_coord)
    if not coords:
        raise ValueError(
            f"Z coordinate '{z_coord}' not found in dataset. "
            f"Available coords: {list(ds.coords.keys())}"
        )
    sliced = ds
    for c in coords:
        sliced = sliced.sel({c: slice(None, z_max)})
        if sliced.sizes[c] == 0:
            raise ValueError(
                f"No data found with {c} <= {z_max}. "
                f"Available range: {ds[c].min().item():.1f}"
                f" to {ds[c].max().item():.1f}"
            )
    return sliced


def _process_file(
    src_path: Path,
    out_dir:  Path,
    cfg:      Config,
    dry_run:  bool,
    log:      logging.Logger,
) -> bool:
    """Process one file. Returns True on success."""
    step_cfg = cfg.steps.splitz
    z_max    = step_cfg.z_max
    z_coord  = step_cfg.z_coord

    # splitz only applies to 3D files
    if "_av_3d" not in src_path.stem:
        log.debug("[splitz] Skipping non-3D file: %s", src_path.name)
        return True

    chunks = try_dask_chunks(_CHUNKS_2D, _CHUNKS_3D, is_3d=True)
    ds     = open_dataset(src_path, chunks)

    if z_coord not in ds.coords:
        log.warning("[splitz] Z coordinate '%s' not found in %s — skipping.",
                    z_coord, src_path.name)
        ds.close()
        return True

    z_min_avail = ds[z_coord].min().item()
    z_max_avail = ds[z_coord].max().item()
    n_z_total   = ds.sizes[z_coord]
    n_z_out     = int((ds[z_coord] <= z_max).sum().item())

    all_z = z_coords_present(ds, z_coord)
    log.debug("[splitz] %s: z range %.1f–%.1f m, z_max=%.1f → %d/%d levels "
              "(coords sliced: %s)",
              src_path.name, z_min_avail, z_max_avail, z_max, n_z_out,
              n_z_total, ", ".join(all_z))

    if z_max > z_max_avail:
        log.warning("[splitz] z_max (%.1f) exceeds file max (%.1f) in %s — all z levels kept.",
                    z_max, z_max_avail, src_path.name)
    if z_max < z_min_avail:
        log.error("[splitz] z_max (%.1f) is below minimum z (%.1f) in %s — skipping.",
                  z_max, z_min_avail, src_path.name)
        ds.close()
        return False

    out_path = out_dir / src_path.name
    if not should_write(out_path, cfg.overwrite, log):
        ds.close()
        return True

    if dry_run:
        log.info("[splitz] [DRY RUN] Would slice %s → %d/%d z levels → %s",
                 src_path.name, n_z_out, n_z_total, out_path.name)
        ds.close()
        return True

    log.info("[splitz] Slicing %s (z <= %.1f m, %d/%d levels) → %s ...",
             src_path.name, z_max, n_z_out, n_z_total, out_path.name)
    t0 = time.monotonic()
    try:
        ds_sliced = _slice_by_z(ds, z_max, z_coord)
        write_dataset(ds_sliced, out_path, cfg.complevel)
        log.info("[splitz]   ✓  %s  (%s)", fmt_size(out_path), fmt_elapsed(t0))
        return True
    except Exception as exc:
        log.error("[splitz] FAILED writing %s: %s", out_path.name, exc)
        return False
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(cfg: Config, dry_run: bool, log: logging.Logger) -> None:
    """
    Run the splitz step for all .nc files in cfg.paths.output_splitvar.
    Raises RuntimeError if any file fails.
    """
    step(log, "Restricting 3D files to z_max")

    input_dir = cfg.paths.output_splitvar
    out_dir   = cfg.paths.output_splitz

    nc_files = sorted(input_dir.glob("*.nc"))
    if not nc_files:
        log.warning("[splitz] No .nc files found in %s", input_dir)
        return

    # Filter to 3D files only for logging purposes
    files_3d = [f for f in nc_files if "_av_3d" in f.stem]
    log.info("[splitz] %d 3D file(s) to process (out of %d total)",
             len(files_3d), len(nc_files))

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for src_path in nc_files:
        if not _process_file(src_path, out_dir, cfg, dry_run, log):
            failures += 1

    if failures:
        raise RuntimeError(f"[splitz] {failures} file(s) failed — check log above.")

    log.info("[splitz] Done.")
