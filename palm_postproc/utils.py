"""
palm_postproc.utils
-------------------
Shared helpers used by all step modules.
"""

from __future__ import annotations

import datetime
import shlex
import sys
import time
from pathlib import Path

import xarray as xr


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def provenance_attrs(cfg, cfg_hash: str, src_path: Path | None = None) -> dict:
    """Global attributes recording HOW an output file was produced.

    Six months later, an OUTPUT_coord directory is otherwise anonymous: the
    filename says which variable it holds and nothing about which config,
    which code, or which source file produced it. These attributes make each
    file self-describing, and `palm_postproc_config_sha` in particular lets
    you tell two runs apart when only a setting changed.

    Written in CF style: `history` is appended to, never replaced, so a file
    that already carries PALM's own history keeps it.
    """
    from . import __version__

    stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    cmd = " ".join(shlex.quote(a) for a in sys.argv)
    out = {
        "palm_postproc_version":    __version__,
        "palm_postproc_config":     str(getattr(cfg, "_source", "") or ""),
        "palm_postproc_config_sha": cfg_hash,
        "palm_postproc_case":       cfg.case,
        "palm_postproc_command":    cmd,
        "palm_postproc_processed":  stamp,
    }
    if src_path is not None:
        out["palm_postproc_source"] = str(src_path)
    return out


def stamp_provenance(ds: xr.Dataset, cfg, cfg_hash: str,
                     src_path: Path | None = None,
                     stage: str | None = None) -> xr.Dataset:
    """Attach provenance attributes to *ds*, appending to `history`.

    `stage` names the last step that produced the file; see set_stage.
    """
    from . import __version__

    attrs = provenance_attrs(cfg, cfg_hash, src_path)
    if stage:
        set_stage(ds, stage)
    stamp = attrs["palm_postproc_processed"]
    line = f"{stamp}: palm_postproc {__version__}: {attrs['palm_postproc_command']}"
    existing = str(ds.attrs.get("history", "")).rstrip()
    ds.attrs.update(attrs)
    ds.attrs["history"] = f"{existing}\n{line}".strip() if existing else line
    return ds


def set_stage(ds: xr.Dataset, stage: str) -> None:
    """Mark *ds* with the step that wrote it (`palm_postproc_stage`).

    Join stamps "join" and palm2gis reads that as confirmation that a file
    is its intended input. Every later step copies the global attributes
    along, so without this an analysis file - origin and units already
    applied - still claimed to be join output.
    """
    ds.attrs["palm_postproc_stage"] = stage


# ---------------------------------------------------------------------------
# NetCDF I/O
# ---------------------------------------------------------------------------

def make_encoding(ds: xr.Dataset, complevel: int) -> dict:
    """zlib encoding dict for all data vars + time coordinate."""
    enc = {
        v: {"zlib": True, "complevel": complevel, "shuffle": True}
        for v in ds.data_vars
    }
    if "time" in ds.coords:
        enc["time"] = {"dtype": "float64"}
    return enc


def write_dataset(ds_out: xr.Dataset, out_path: Path, complevel: int,
                  stage: str | None = None) -> None:
    """Write dataset to NetCDF4 with zlib compression, marked with *stage*."""
    if stage:
        ds_out = ds_out.copy()
        set_stage(ds_out, stage)
    encoding = make_encoding(ds_out, complevel)
    # A time-invariant variable (ind_z_xy, zusi, zwwi) selected out of a
    # larger dataset inherits that dataset's encoding, which still declares
    # time unlimited. to_netcdf then warns on every such file.
    if not any("time" in ds_out[v].dims for v in ds_out.data_vars):
        ds_out.encoding.pop("unlimited_dims", None)
    ds_out.to_netcdf(out_path, format="NETCDF4", encoding=encoding)


def open_dataset(src_path: Path, chunks: dict | None) -> xr.Dataset:
    """Open a PALM NetCDF file with decode_times=False, decode_cf=False."""
    return xr.open_dataset(
        src_path, engine="netcdf4", chunks=chunks,
        decode_times=False, decode_cf=False,
    )


def try_dask_chunks(chunks_2d: dict, chunks_3d: dict, is_3d: bool,
                    src_path: Path | None = None) -> dict | None:
    """Return chunk dict if dask is available, else None.

    With src_path the chunks are aligned to how the file is stored, see
    _align_chunks.
    """
    try:
        import dask  # noqa: F401
    except ImportError:
        return None
    chunks = chunks_3d if is_3d else chunks_2d
    if src_path is None:
        return chunks
    return _align_chunks(chunks, src_path)


def _align_chunks(chunks: dict, src_path: Path) -> dict:
    """Fit the requested dask chunks to the file's stored chunks.

    A joined file stores each 3D variable in compressed chunks that span
    the whole column or more (netCDF-C default chunking, e.g. [1, 80, 200,
    200]). Asking dask for zu_3d=10 cut every stored chunk into several
    reads, and each read decompresses the whole chunk again: correct but
    several times slower, and xarray warns on every file. So each requested
    size is rounded up to a whole number of stored chunks, and a dimension
    stored in pieces smaller than its length is chunked as stored, which
    keeps a dask chunk the size of a few stored ones.

    Only the requested dimensions are set. Variables of different shape
    store a dimension in different pieces - a 3D field on zu_3d next to a
    soil field on zs_3d can hold y in chunks of 334 and 500 - and any one
    size set for y cuts some variable's chunks (xarray: 'separate the
    stored chunks along dimension "y" starting at index 500'). A dimension
    left out follows each variable's own stored chunks. A requested one is
    rounded to a common multiple of every stored size along it, or read
    whole when no such multiple fits.
    """
    from math import lcm

    from netCDF4 import Dataset

    stored: dict = {}
    sizes:  dict = {}
    with Dataset(src_path, "r") as nc:
        for dn, d in nc.dimensions.items():
            sizes[dn] = len(d)
        for v in nc.variables.values():
            layout = v.chunking()
            if layout == "contiguous" or layout is None:
                continue
            for dn, c in zip(v.dimensions, layout):
                if dn in chunks:
                    stored.setdefault(dn, set()).add(int(c))

    out = {}
    for dn, n in chunks.items():
        if dn not in stored:
            out[dn] = n             # not chunked in the file: keep the request
            continue
        step = lcm(*stored[dn])
        n = -(-int(n) // step) * step                   # ceil to a multiple
        # time is unlimited and grows; any other dimension that a common
        # multiple does not fit into is read whole
        out[dn] = n if dn == "time" or n < sizes.get(dn, 0) else -1
    return out


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt_size(path: Path) -> str:
    mb = path.stat().st_size / 1_048_576
    return f"{mb:.1f} MB"


def fmt_elapsed(t0: float) -> str:
    return f"{time.monotonic() - t0:.1f}s"


def fmt_duration(secs: float) -> str:
    if secs < 60:
        return f"{secs:.0f}s"
    elif secs < 3600:
        return f"{secs / 60:.1f}m"
    elif secs < 86400:
        return f"{secs / 3600:.1f}h"
    else:
        return f"{secs / 86400:.1f}d"


# ---------------------------------------------------------------------------
# Overwrite guard
# ---------------------------------------------------------------------------

def should_write(out_path: Path, overwrite: bool, log, tag: str = "") -> bool:
    """
    Return True if the file should be written.
    If overwrite=True, always True.
    If overwrite=False and file exists, log "kept (exists)" and return False
    (the same wording palm2gis uses; `overwrite: true` replaces it).
    """
    label = f"[{tag}] " if tag else ""
    if not out_path.exists():
        return True
    if overwrite:
        log.debug("%sreplacing %s", label, out_path.name)
        return True
    log.info("%skept (exists): %s", label, out_path.name)
    return False
