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
                     src_path: Path | None = None) -> xr.Dataset:
    """Attach provenance attributes to *ds*, appending to `history`."""
    from . import __version__

    attrs = provenance_attrs(cfg, cfg_hash, src_path)
    stamp = attrs["palm_postproc_processed"]
    line = f"{stamp}: palm_postproc {__version__}: {attrs['palm_postproc_command']}"
    existing = str(ds.attrs.get("history", "")).rstrip()
    ds.attrs.update(attrs)
    ds.attrs["history"] = f"{existing}\n{line}".strip() if existing else line
    return ds


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


def write_dataset(ds_out: xr.Dataset, out_path: Path, complevel: int) -> None:
    """Write dataset to NetCDF4 with zlib compression."""
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


def try_dask_chunks(chunks_2d: dict, chunks_3d: dict, is_3d: bool) -> dict | None:
    """Return chunk dict if dask is available, else None."""
    try:
        import dask  # noqa: F401
        return chunks_3d if is_3d else chunks_2d
    except ImportError:
        return None


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

def should_write(out_path: Path, overwrite: bool, log) -> bool:
    """
    Return True if the file should be written.
    If overwrite=True, always True.
    If overwrite=False and file exists, log a warning and return False.
    """
    if not out_path.exists():
        return True
    if overwrite:
        log.debug("Overwriting existing file: %s", out_path.name)
        return True
    log.warning("Output already exists, skipping (set overwrite: true to replace): %s",
                out_path.name)
    return False
