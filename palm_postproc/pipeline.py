"""
palm_postproc.pipeline
----------------------
Orchestrates the five processing steps in order:
  join → splitvar → splitz → splittime → coord

Two modes, selected by cfg.chain:

  chain=True  (default)
    join runs first and writes OUTPUT_join (always — it cannot be chained
    since it uses netCDF4 directly). Then each joined file is passed through
    splitvar → splitz → splittime → coord in memory; only the final result
    is written to disk.

  chain=False
    Classic step-by-step mode: each step reads from the previous step's
    output directory and writes its own. Useful for debugging.

Parallelism
-----------
If cfg.workers > 1, the post-join steps process files in parallel using
ProcessPoolExecutor. Each worker handles one source file independently.

Resume / skip
-------------
A state file ({base}/{case}/.palm_postproc_state.json) records a fingerprint
for every output file. Files whose source and config are unchanged are
skipped automatically, even across separate runs.

--only
------
Passed as a set of step names. When set, only those steps run
(chain mode is forced off; intermediate dirs must already exist).
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import xarray as xr

from .config import Config
from .log import step
from .state  import State, config_hash, join_config_hash

log = logging.getLogger("palm_postproc")

_ALL_STEPS = ("join", "splitvar", "splitz", "splittime", "coord")


# ---------------------------------------------------------------------------
# Step-enabled helper
# ---------------------------------------------------------------------------

def _step_enabled(name: str, cfg: Config, only: Optional[set]) -> bool:
    step = getattr(cfg.steps, name)
    if not step.enabled:
        return False
    if only is not None and name not in only:
        return False
    return True


# ---------------------------------------------------------------------------
# In-memory transform functions (chained pipeline)
# ---------------------------------------------------------------------------

def _apply_splitvar(
    ds: xr.Dataset, src_path: Path, cfg: Config
) -> list[tuple[str, xr.Dataset]]:
    """Return list of (suffix, dataset) pairs — one per variable / group."""
    from .steps.splitvar import _detect_mode, _resolve_vars

    stem  = src_path.stem
    mode  = _detect_mode(stem)
    if mode is None:
        # File type other than _av_xy / _av_3d (e.g. _pr, _surf, _topo_surf,
        # _ts) — no per-variable whitelist applies; pass the whole file
        # through unmodified to the next step.
        log.debug("[pipeline] %s is not _av_xy/_av_3d — no variable splitting applied.",
                  src_path.name)
        return [("", ds)]

    is_3d     = (mode == "3D")
    step_cfg  = cfg.steps.splitvar
    requested = step_cfg.vars_3d if is_3d else step_cfg.vars_2d
    data_vars = _resolve_vars(ds, requested, log, label=mode)

    if not data_vars:
        return []

    group_vars      = [v for v in step_cfg.group_vars if v in data_vars] if is_3d else []
    individual_vars = [v for v in data_vars if v not in group_vars]

    results = []
    if group_vars:
        results.append((f".{step_cfg.group_suffix}", ds[group_vars]))
    for varname in individual_vars:
        safe = varname.replace("*", "")
        results.append((f".{safe}", ds[[varname]]))
    return results


def _apply_splitz(ds: xr.Dataset, cfg: Config) -> xr.Dataset:
    from .steps.splitz import _slice_by_z, z_coords_present
    step_cfg = cfg.steps.splitz
    # Any known vertical coordinate is enough: a `.uvw` group file carries
    # w on zw_3d, which must be sliced even when zu_3d is the configured one.
    if not z_coords_present(ds, step_cfg.z_coord):
        return ds
    return _slice_by_z(ds, step_cfg.z_max, step_cfg.z_coord)


def _apply_splittime(ds: xr.Dataset, cfg: Config) -> xr.Dataset:
    from .steps.splittime import _select_timesteps, _parse_duration
    step_cfg = cfg.steps.splittime
    if "time" not in ds.dims:
        return ds
    n_time        = ds.sizes["time"]
    t_start       = step_cfg.t_start if step_cfg.t_start is not None else 0
    t_end         = step_cfg.t_end   if step_cfg.t_end   is not None else n_time - 1
    timestep_secs = _parse_duration(step_cfg.timestep) if step_cfg.timestep else None
    return _select_timesteps(ds, t_start, t_end, timestep_secs)


def _apply_coord(ds: xr.Dataset, src_path: Path, cfg: Config) -> xr.Dataset:
    from .steps.coord import _add_coordinates
    return _add_coordinates(ds, cfg.steps.coord, src_path, log)


def _has_time(ds: xr.Dataset) -> bool:
    """
    True only if 'time' is an actual dimension of at least one data variable
    — not merely a leftover coordinate carried over from the parent dataset
    (which happens for time-invariant variables like ind_z_xy, zusi, zwwi
    after splitvar selects them out of a larger dataset).
    """
    return any("time" in ds[v].dims for v in ds.data_vars)


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------

def _final_output_dir(cfg: Config, only: Optional[set]) -> Path:
    """Return the output directory of the last enabled post-join step."""
    steps_dirs = [
        ("splitvar",  cfg.paths.output_splitvar),
        ("splitz",    cfg.paths.output_splitz),
        ("splittime", cfg.paths.output_splittime),
        ("coord",     cfg.paths.output_coord),
    ]
    enabled = [(name, d) for name, d in steps_dirs
               if _step_enabled(name, cfg, only)]
    if not enabled:
        # No post-join steps enabled — output stays in output_join
        return cfg.paths.output_join
    return enabled[-1][1]


def _output_stem(src_stem: str, var_suffix: str,
                 cfg: Config, only: Optional[set], has_time: bool) -> str:
    """Build the final output filename stem."""
    stem = src_stem + var_suffix
    # Only append .utm when coord actually ran (time-invariant files skip it)
    if _step_enabled("coord", cfg, only) and has_time:
        stem += ".utm"
    return stem


# ---------------------------------------------------------------------------
# Chained single-file processor
# ---------------------------------------------------------------------------

def _process_file_chained(
    src_path: Path,
    cfg:      Config,
    state:    State,
    cfg_h:    str,
    dry_run:  bool,
    only:     Optional[set],
    defer_save: bool = False,
) -> int:
    """Process one file through all enabled post-join steps. Returns failure count.

    With defer_save the State is updated in memory but never written to
    disk — used by worker processes, whose records the parent merges and
    saves once (concurrent saves would clobber each other).
    """
    from .utils import (try_dask_chunks, fmt_size, fmt_elapsed,
                        stamp_provenance)

    _CHUNKS_2D = {"time": 4}
    _CHUNKS_3D = {"time": 4, "zu_3d": 10}

    stem  = src_path.stem
    is_3d = "_av_3d" in stem

    out_dir = _final_output_dir(cfg, only)
    chunks  = try_dask_chunks(_CHUNKS_2D, _CHUNKS_3D, is_3d)

    # --- Peek pass: determine what outputs this file will produce -----------
    ds_peek = xr.open_dataset(src_path, engine="netcdf4", chunks=chunks,
                               decode_times=False, decode_cf=False)

    if _step_enabled("splitvar", cfg, only):
        splits_peek = _apply_splitvar(ds_peek, src_path, cfg)
    else:
        splits_peek = [("", ds_peek)]

    ds_peek.close()

    if not splits_peek:
        log.warning("[pipeline] No variables to process in %s", src_path.name)
        return 0

    # Build (suffix, has_time, out_path) for each output
    all_outputs = []
    for var_suffix, ds_var in splits_peek:
        ht       = _has_time(ds_var)
        out_stem = _output_stem(stem, var_suffix, cfg, only, ht)
        out_path = out_dir / f"{out_stem}.nc"
        all_outputs.append((var_suffix, ht, out_path))

    # Resume check
    outputs_needed = []
    for var_suffix, ht, out_path in all_outputs:
        if state.is_current(src_path, out_path, cfg_h):
            log.info("[pipeline] Up to date, skipping: %s", out_path.name)
        else:
            outputs_needed.append((var_suffix, ht, out_path))

    if not outputs_needed:
        return 0

    if dry_run:
        for _, _, out_path in outputs_needed:
            log.info("[pipeline] [DRY RUN] Would write: %s", out_path.name)
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Processing pass: open once, apply all steps -----------------------
    ds_full = xr.open_dataset(src_path, engine="netcdf4", chunks=chunks,
                               decode_times=False, decode_cf=False)

    splits_full = (
        _apply_splitvar(ds_full, src_path, cfg)
        if _step_enabled("splitvar", cfg, only)
        else [("", ds_full)]
    )

    failures       = 0
    needed_suffixes = {var_suffix for var_suffix, _, _ in outputs_needed}
    # Build lookup: var_suffix → (ht, out_path)
    output_map     = {var_suffix: (ht, out_path)
                      for var_suffix, ht, out_path in outputs_needed}

    for var_suffix, ds_var in splits_full:
        if var_suffix not in needed_suffixes:
            continue

        ht, out_path = output_map[var_suffix]

        log.info("[pipeline] Processing %s → %s ...", src_path.name, out_path.name)
        t0 = time.monotonic()

        try:
            ds_work = ds_var

            if _step_enabled("splitz", cfg, only) and is_3d:
                ds_work = _apply_splitz(ds_work, cfg)

            if _step_enabled("splittime", cfg, only) and _has_time(ds_work):
                ds_work = _apply_splittime(ds_work, cfg)

            if _step_enabled("coord", cfg, only):
                if not _has_time(ds_work):
                    log.debug("[pipeline] Skipping coord for time-invariant: %s",
                              out_path.name)
                else:
                    ds_work = _apply_coord(ds_work, src_path, cfg)

            # Drop time-invariant datasets' lingering 'time' coordinate before
            # writing, so the file does not declare an unlimited 'time'
            # dimension that no variable actually uses (which produces
            # near-empty/corrupt output for vars like ind_z_xy, zusi, zwwi).
            time_is_used_dim = any(
                "time" in ds_work[v].dims for v in ds_work.data_vars
            )
            if "time" in ds_work.coords and not time_is_used_dim:
                ds_work = ds_work.drop_vars("time")
            if not time_is_used_dim:
                # The source dataset's encoding still declares time as an
                # unlimited dimension. Carrying that into to_netcdf() for a
                # dataset that no longer has a time dimension raises a
                # UserWarning on every time-invariant variable written.
                ds_work.encoding.pop("unlimited_dims", None)

            ds_work = stamp_provenance(ds_work, cfg, cfg_h, src_path)

            encoding = {
                v: {"zlib": True, "complevel": cfg.complevel, "shuffle": True}
                for v in ds_work.data_vars
            }
            if time_is_used_dim and "time" in ds_work.coords:
                encoding["time"] = {"dtype": "float64"}
            ds_work.to_netcdf(out_path, format="NETCDF4", encoding=encoding)

            log.info("[pipeline]   ✓  %s  (%s)", fmt_size(out_path), fmt_elapsed(t0))
            state.mark_done(src_path, out_path, cfg_h)
            if not defer_save:
                state.save()

        except Exception as exc:
            log.error("[pipeline] FAILED %s: %s", out_path.name, exc)
            log.debug("[pipeline]", exc_info=True)
            state.invalidate(out_path)
            if not defer_save:
                state.save()
            failures += 1

    ds_full.close()
    return failures


# ---------------------------------------------------------------------------
# Classic step-by-step pipeline (chain=False or --only)
# ---------------------------------------------------------------------------

def _run_classic(cfg: Config, dry_run: bool, only: Optional[set]) -> None:
    from .steps import join, splitvar, splitz, splittime, coord

    step_fns = [
        ("join",      cfg.steps.join,      join.run),
        ("splitvar",  cfg.steps.splitvar,  splitvar.run),
        ("splitz",    cfg.steps.splitz,    splitz.run),
        ("splittime", cfg.steps.splittime, splittime.run),
        ("coord",     cfg.steps.coord,     coord.run),
    ]

    for name, step_cfg, run_fn in step_fns:
        if not _step_enabled(name, cfg, only):
            log.info("step %s: %s", name,
                     "disabled" if not step_cfg.enabled
                     else "skipped (--only)")
            continue
        run_fn(cfg, dry_run=dry_run, log=log)


# ---------------------------------------------------------------------------
# Worker for parallel execution (top-level for pickling)
# ---------------------------------------------------------------------------

def _worker(args: tuple) -> tuple:
    """
    Run one file in a subprocess.

    Returns (failures, records) rather than saving: each worker holds its
    own State, so concurrent save() calls would overwrite each other and
    lose resume information. The parent merges and saves once.
    """
    src_path, cfg, cfg_h, dry_run, only = args
    from .state import State
    state = State.load(cfg)
    before = set(state.records())
    failures = _process_file_chained(src_path, cfg, state, cfg_h, dry_run,
                                     only, defer_save=True)
    records = {k: v for k, v in state.records().items() if k not in before}
    return failures, records


# ---------------------------------------------------------------------------
# Determine which directory feeds the post-join chained steps
# ---------------------------------------------------------------------------

def _chained_input_dir(cfg: Config) -> Path:
    """
    Determine which directory feeds the post-join chained steps:
      1. paths.input (explicit override) — bypasses join entirely
      2. paths.output_join              — if join is enabled
      3. paths.join_input               — if join is disabled (pre-joined files)
    """
    if cfg.paths.input_override is not None:
        return cfg.paths.input_override
    if cfg.steps.join.enabled:
        return cfg.paths.output_join
    return cfg.paths.join_input


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    cfg:     Config,
    dry_run: bool = False,
    only:    Optional[set] = None,
) -> None:
    if only is not None:
        step(log, "Running selected steps: {}", ", ".join(sorted(only)))
        _run_classic(cfg, dry_run, only)
        return

    if not cfg.chain:
        step(log, "Running the classic step-by-step pipeline")
        _run_classic(cfg, dry_run, only=None)
        return

    # ---- Join step (always classic — uses netCDF4 directly) ---------------
    if _step_enabled("join", cfg, only=None):
        from .steps.join import run as join_run
        # join outputs are guarded only by a file-exists check, so a change
        # to the join settings would otherwise leave stale files in place.
        _jstate = State.load(cfg)
        _jhash  = join_config_hash(cfg)
        _prev   = _jstate.get_meta("join_config_hash")
        _force  = _prev is not None and _prev != _jhash
        if _force:
            log.warning("[pipeline] join settings changed since the last run "
                        "(%s → %s) — rejoining, existing files will be "
                        "replaced.", _prev, _jhash)
        join_run(cfg, dry_run=dry_run, log=log, force=_force)
        if not dry_run:
            _jstate.set_meta("join_config_hash", _jhash)
            _jstate.save()
    else:
        log.info("step join: disabled")

    # ---- Chained post-join steps ------------------------------------------
    post_join_enabled = any(
        _step_enabled(n, cfg, only=None)
        for n in ("splitvar", "splitz", "splittime", "coord")
    )
    if not post_join_enabled:
        log.info("[pipeline] No post-join steps enabled — done.")
        return

    step(log, "Processing each file through all steps in memory")
    if cfg.workers > 1:
        log.info("Parallel workers: %d", cfg.workers)

    state    = State.load(cfg)
    cfg_h    = config_hash(cfg)
    log.debug("[pipeline] Config hash: %s", cfg_h)

    input_dir = _chained_input_dir(cfg)
    all_files = sorted(input_dir.glob("*.nc"))

    # Only _av_xy / _av_3d files go through the post-join pipeline.
    # Other PALM output types (_pr, _surf, _topo_surf, _ts, ...) are joined
    # but not further processed — they have no splitvar/splitz/coord meaning.
    from .steps.splitvar import _detect_mode
    src_files = [f for f in all_files if _detect_mode(f.stem) is not None]
    skipped   = [f for f in all_files if _detect_mode(f.stem) is None]

    if skipped:
        log.info("[pipeline] %d file(s) not _av_xy/_av_3d — left as-is in %s:",
                 len(skipped), input_dir.name)
        for f in skipped:
            log.debug("[pipeline]   %s", f.name)

    if not src_files:
        log.warning("[pipeline] No _av_xy/_av_3d files found in %s", input_dir)
        return

    log.info("[pipeline] %d source file(s) to process from %s",
             len(src_files), input_dir)
    t_total        = time.monotonic()
    total_failures = 0

    if cfg.workers > 1 and not dry_run:
        args_list = [
            (f, cfg, cfg_h, dry_run, None)
            for f in src_files
        ]
        with ProcessPoolExecutor(max_workers=cfg.workers) as executor:
            futures = {executor.submit(_worker, a): a[0] for a in args_list}
            for future in as_completed(futures):
                src = futures[future]
                try:
                    failures, records = future.result()
                    total_failures += failures
                    # Merge in the parent: workers never write the state file.
                    state.merge(records)
                    state.save()
                except Exception as exc:
                    log.error("[pipeline] Worker crashed for %s: %s", src.name, exc)
                    total_failures += 1
    else:
        for src_path in src_files:
            total_failures += _process_file_chained(
                src_path, cfg, state, cfg_h, dry_run, only=None
            )

    elapsed = time.monotonic() - t_total
    if total_failures:
        raise RuntimeError(
            f"[pipeline] {total_failures} file(s) failed in {elapsed:.1f}s"
            " — check log above."
        )
    log.info("[pipeline] All files done in %.1fs.", elapsed)
