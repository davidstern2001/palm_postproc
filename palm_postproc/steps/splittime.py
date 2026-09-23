"""
palm_postproc.steps.splittime
------------------------------
Subsamples or slices the time dimension of PALM output files.
Refactored from palm_splittime.py — logic unchanged.
"""

from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

from ..config import Config, SplittimeConfig
from ..log import step
from ..timespec import TimeSpecError, parse_origin, to_seconds
from ..utils import (
    open_dataset, write_dataset, try_dask_chunks,
    fmt_size, fmt_elapsed, fmt_duration, should_write,
)

_CHUNKS_2D = {"time": 1}
_CHUNKS_3D = {"time": 1, "zu_3d": 10}


# ---------------------------------------------------------------------------
# Core logic (unchanged from palm_splittime.py)
# ---------------------------------------------------------------------------

def _parse_duration(s: str) -> float:
    """Parse a duration string (e.g. '1h', '30m', '3600') to seconds."""
    s = s.strip().lower()
    try:
        return float(s)
    except ValueError:
        pass
    suffixes = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if s[-1] in suffixes:
        return float(s[:-1]) * suffixes[s[-1]]
    raise ValueError(f"Invalid duration '{s}'. Use plain seconds or a suffix: s/m/h/d.")


def _window_indices(ds, step_cfg, src_path, log):
    """The [first, last] record of the configured window.

    time.from / time.to are resolved against this file's own axis, so one
    window covers the same period in files of different output interval.
    The pre-0.6 record indices (t_start / t_end) are used when no window
    is given.
    """
    n_time = ds.sizes["time"]
    if step_cfg.time_from is None and step_cfg.time_to is None:
        t_start = step_cfg.t_start if step_cfg.t_start is not None else 0
        t_end = step_cfg.t_end if step_cfg.t_end is not None else n_time - 1
        return t_start, t_end

    origin = parse_origin(ds.attrs.get("origin_time"))
    t_values = ds["time"].values.astype(float)
    t_from = to_seconds(step_cfg.time_from, origin)
    t_to = to_seconds(step_cfg.time_to, origin)
    keep = np.ones(n_time, dtype=bool)
    if t_from is not None:
        keep &= t_values >= t_from - 1e-6
    if t_to is not None:
        keep &= t_values <= t_to + 1e-6
    if not keep.any():
        raise ValueError(
            f"no record of {src_path.name} falls in the time window "
            f"({t_values[0]:.0f} .. {t_values[-1]:.0f} s in the file)")
    idx = np.nonzero(keep)[0]
    log.debug("[splittime] %s: window -> records %d..%d of %d",
              src_path.name, int(idx[0]), int(idx[-1]), n_time)
    return int(idx[0]), int(idx[-1])


def _select_timesteps(
    ds:            xr.Dataset,
    t_start:       int,
    t_end:         int,
    timestep_secs: Optional[float],
) -> xr.Dataset:
    """Return dataset sliced to [t_start, t_end] and optionally subsampled."""
    n_time = ds.sizes["time"]
    if not (0 <= t_start <= t_end < n_time):
        raise ValueError(
            f"Invalid time range [{t_start}, {t_end}] for dataset with {n_time} timesteps."
        )

    t_values = ds["time"].values.astype(float)

    if timestep_secs is None:
        out = ds.isel(time=slice(t_start, t_end + 1))
        return _stamp_time_window(out, ds, t_start, t_end, None)

    t0_val  = t_values[t_start]
    indices = []
    for i in range(t_start, t_end + 1):
        elapsed = t_values[i] - t0_val
        n       = round(elapsed / timestep_secs)
        if abs(t_values[i] - (t0_val + n * timestep_secs)) < timestep_secs * 0.01:
            indices.append(i)

    if not indices:
        raise ValueError(
            f"No timesteps match interval {timestep_secs:.0f}s in range [{t_start}, {t_end}]."
        )
    return _stamp_time_window(ds.isel(time=indices), ds,
                              t_start, t_end, timestep_secs)


def _stamp_time_window(out, src, t_start: int, t_end: int,
                       timestep_secs: Optional[float]):
    """Record which time window was kept, for the same reason as splitz.

    A time-sliced file looks exactly like a full one, so anything that
    resolves a window against the file's own axis gets a different answer
    depending on whether this step already ran.
    """
    out.attrs["palm_postproc_time_start"] = int(t_start)
    out.attrs["palm_postproc_time_end"] = int(t_end)
    out.attrs["palm_postproc_time_records"] = f"{src.sizes['time']}->{out.sizes['time']}"
    if timestep_secs is not None:
        out.attrs["palm_postproc_timestep"] = float(timestep_secs)
    if out.sizes["time"]:
        tv = out["time"].values
        out.attrs["palm_postproc_time_range"] = f"{float(tv[0]):.1f},{float(tv[-1]):.1f}"
    return out


def _process_file(
    src_path:  Path,
    out_dir:   Path,
    cfg:       Config,
    dry_run:   bool,
    log:       logging.Logger,
) -> bool:
    """Process one file. Returns True on success."""
    step_cfg: SplittimeConfig = cfg.steps.splittime

    is_3d  = "_av_3d" in src_path.stem
    chunks = try_dask_chunks(_CHUNKS_2D, _CHUNKS_3D, is_3d)
    ds     = open_dataset(src_path, chunks)

    if "time" not in ds.dims:
        log.warning("[splittime] no time dimension in %s - skipped.",
                    src_path.name)
        ds.close()
        return True

    n_time   = ds.sizes["time"]
    t_values = ds["time"].values.astype(float)

    try:
        t_start, t_end = _window_indices(ds, step_cfg, src_path, log)
    except (TimeSpecError, ValueError) as exc:
        log.error("[splittime] %s", exc)
        ds.close()
        return False

    # Validate
    if not (0 <= t_start < n_time):
        log.error("[splittime] t_start %d out of range [0, %d] in %s",
                  t_start, n_time - 1, src_path.name)
        ds.close()
        return False
    if not (0 <= t_end < n_time):
        log.error("[splittime] t_end %d out of range [0, %d] in %s",
                  t_end, n_time - 1, src_path.name)
        ds.close()
        return False
    if t_start > t_end:
        log.error("[splittime] t_start (%d) > t_end (%d) in %s",
                  t_start, t_end, src_path.name)
        ds.close()
        return False

    # Parse timestep
    timestep_secs: Optional[float] = None
    if step_cfg.timestep:
        try:
            timestep_secs = _parse_duration(step_cfg.timestep)
        except ValueError as exc:
            log.error("[splittime] %s", exc)
            ds.close()
            return False

    dt_vals = np.diff(t_values)
    min_dt  = float(np.min(dt_vals)) if len(dt_vals) > 0 else 0.0

    if timestep_secs is not None and timestep_secs < min_dt:
        log.warning(
            "[splittime] interval %s is below the file's timestep %s in %s "
            "- few or no timesteps may match.",
            fmt_duration(timestep_secs), fmt_duration(min_dt), src_path.name,
        )

    t_range_s = t_values[t_end] - t_values[t_start]
    n_out = (max(1, round(t_range_s / timestep_secs) + 1)
             if timestep_secs is not None else t_end - t_start + 1)

    log.debug("[splittime] %s: t=[%d,%d], min_dt=%s, interval=%s, ~%d timesteps out",
              src_path.name, t_start, t_end,
              fmt_duration(min_dt),
              fmt_duration(timestep_secs) if timestep_secs else "all",
              n_out)

    out_path = out_dir / src_path.name
    if not should_write(out_path, cfg.overwrite, log, "splittime"):
        ds.close()
        return True

    if dry_run:
        log.info("[splittime] would write %s: ~%d timestep(s) (dry run)",
                 out_path.name, n_out)
        ds.close()
        return True

    t0 = time.monotonic()
    try:
        ds_sliced = _select_timesteps(ds, t_start, t_end, timestep_secs)
        write_dataset(ds_sliced, out_path, cfg.complevel)
        log.info("[splittime] wrote %s: %s in %s", out_path.name,
                 fmt_size(out_path), fmt_elapsed(t0))
        return True
    except Exception as exc:
        log.error("[splittime] %s failed: %s", out_path.name, exc)
        return False
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(cfg: Config, dry_run: bool, log: logging.Logger) -> None:
    """
    Run the splittime step.

    Input is taken from:
      - output_splitz   for 3D files (already z-sliced)
      - output_splitvar for 2D files (splitz does not touch them)

    Raises RuntimeError if any file fails.
    """
    step(log, "Subsampling the time axis")

    out_dir  = cfg.paths.output_splittime
    failures = 0

    # Collect files from upstream directories.
    # 3D files: come from output_splitz if splitz was enabled, else output_splitvar.
    # 2D files: always come from output_splitvar (splitz never touches them).
    sources: list[tuple[Path, str]] = []

    if cfg.steps.splitz.enabled:
        files_3d = sorted(cfg.paths.output_splitz.glob("*.nc"))
        log.debug("[splittime] 3D source: output_splitz (%d file(s))", len(files_3d))
        sources.extend((f, "3D (post-splitz)") for f in files_3d)
    else:
        # splitz disabled — 3D files are still in output_splitvar
        files_3d = [f for f in sorted(cfg.paths.output_splitvar.glob("*.nc"))
                    if "_av_3d" in f.stem]
        log.debug("[splittime] 3D source: output_splitvar (%d file(s), "
                  "splitz off)", len(files_3d))
        sources.extend((f, "3D (post-splitvar, splitz disabled)") for f in files_3d)

    files_2d = [f for f in sorted(cfg.paths.output_splitvar.glob("*.nc"))
                if "_av_xy" in f.stem]
    log.debug("[splittime] 2D source: output_splitvar (%d file(s))", len(files_2d))
    sources.extend((f, "2D (post-splitvar)") for f in files_2d)

    if not sources:
        log.warning("[splittime] no .nc files in the upstream directories.")
        return

    log.info("[splittime] %d file(s) to process", len(sources))

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    for src_path, label in sources:
        log.debug("[splittime] %s: %s", label, src_path.name)
        if not _process_file(src_path, out_dir, cfg, dry_run, log):
            failures += 1

    if failures:
        raise RuntimeError(f"[splittime] {failures} file(s) failed - see the log above.")

