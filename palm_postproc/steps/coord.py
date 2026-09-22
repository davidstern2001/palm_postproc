"""
palm_postproc.steps.coord
--------------------------
Appends georeferenced UTM coordinates and CRS metadata to PALM output files.
Refactored from palm_output_coordinate_pospr.py — logic unchanged.
"""

from __future__ import annotations

import re
import time
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
import pyproj

from ..config import Config, CoordConfig
from ..log import step
from ..utils import (
    write_dataset, try_dask_chunks,
    fmt_size, fmt_elapsed, should_write,
)

_CHUNKS_2D = {"time": 1}
_CHUNKS_3D = {"time": 1, "zu_3d": 10}

# Suffix appended to output filenames by this step
_TRANSFORMED_SUFFIX = ".utm"

# ---------------------------------------------------------------------------
# Temperature variables
# ---------------------------------------------------------------------------
# Kept deliberately IDENTICAL to palm2gis/steps/thermo.py. The two tools
# are separate packages by design and share no code, so this block is the
# one thing that has to be edited in both places at once; the README
# records that.
#
# TEMP_PREFIXES only IDENTIFIES temperature variables. It does NOT say
# what scale they are on, and assuming kelvin here was a real bug: PALM's
# reference tables give `ta` (and `ta_2m*`) in DEGREES CELSIUS, while
# `theta`, `tsurf*` and `t_surf` are kelvin. Blindly subtracting 273.15
# from an already-Celsius `ta` produced voxels at -253 °C. The scale now
# comes from the `units` attribute; see temperature_scale().
TEMP_PREFIXES = ("theta", "tsurf", "t_surf", "ta")
K0 = 273.15

# Potential temperature is exempt from `celsius`. It is a kelvin-defined
# quantity and "potential temperature in °C" is a confusing thing to hand
# to anyone; `celsius` governs actual temperatures only.
CELSIUS_EXEMPT_PREFIXES = ("theta",)

# Unit strings, lowercased and stripped. The prefix entries matter: PALM
# writes its units into a fixed-length character buffer and real files
# come out carrying "degree_" — truncated mid-word — so an exact-match
# list rejects the very data this is meant to handle.
_KELVIN_UNITS = ("k", "kelvin", "degree_k", "degrees_k", "deg_k")
_CELSIUS_UNITS = ("degc", "deg_c", "degree_c", "degrees_c",
                  "celsius", "\N{DEGREE SIGN}c", "c")
_CELSIUS_PREFIXES = ("degree", "degrees", "deg")

# PALM marks masked/undefined points (inside buildings, above the
# terrain-following domain top) with a sentinel such as -9999.0. Subtracting
# 273.15 from those turns them into nonsense like -1e6, so they are excluded.
_COMMON_FILL_VALUES = (-9999.0, -999999.0, -9999.9)


class TemperatureUnitError(RuntimeError):
    """Raised when a temperature variable's scale cannot be determined."""


def is_temperature(name: str) -> bool:
    """True when *name* is a temperature variable by PALM naming."""
    return str(name).lower().startswith(TEMP_PREFIXES)


def is_celsius_exempt(name: str) -> bool:
    """True when *name* stays in kelvin regardless of the celsius setting."""
    return str(name).lower().startswith(CELSIUS_EXEMPT_PREFIXES)


def temperature_scale(units, name: str = "", override: dict | None = None):
    """Return 'K' or 'C' for a temperature variable.

    Resolution order: explicit config override, then the `units` string.
    There is deliberately NO fallback that guesses from the values. A
    temperature silently off by 273.15 still looks like a temperature,
    which is exactly how the -253 °C voxels survived a release; failing
    loudly is cheaper than the alternative.

    Raises TemperatureUnitError when neither source resolves.
    """
    if override:
        for key, scale in override.items():
            if str(name).lower() == str(key).lower():
                s = str(scale).strip().upper()[:1]
                if s in ("K", "C"):
                    return s
                raise TemperatureUnitError(
                    f"temperature_units override for '{name}' is {scale!r}; "
                    f"expected 'K' or 'C'")

    u = str(units or "").strip().lower().replace(" ", "")
    if u in _KELVIN_UNITS:
        return "K"
    if u in _CELSIUS_UNITS or u.startswith(_CELSIUS_PREFIXES):
        return "C"

    raise TemperatureUnitError(
        f"cannot determine the temperature scale of '{name}': units="
        f"{units!r}. Add an entry to steps.coord.temperature_units "
        f"(e.g. {{{name}: C}}) to state it explicitly.")


def kelvin_to_celsius_da(da: xr.DataArray, fill_val=None) -> xr.DataArray:
    """K -> °C on the VALID entries of *da* only, lazily.

    Fill sentinels and non-finite entries are passed through untouched, so
    NoData stays NoData instead of becoming ~-9726 °C. Implemented with
    xr.where so a dask-backed array stays lazy and is converted chunk by
    chunk when written.
    """
    valid = da.notnull() & xr.apply_ufunc(np.isfinite, da, dask="allowed")
    if fill_val is not None:
        valid = valid & (da != fill_val)
    else:
        for fv in _COMMON_FILL_VALUES:
            valid = valid & (abs(da - fv) > 1e-3)
    return xr.where(valid, da - K0, da)


def celsius_to_kelvin_da(da: xr.DataArray, fill_val=None) -> xr.DataArray:
    """°C -> K on the VALID entries of *da* only, lazily.

    The mirror of kelvin_to_celsius_da, needed now that `celsius: false`
    means "emit kelvin" rather than "do nothing": PALM's own `ta` arrives
    in °C, so producing a kelvin file requires converting it.
    """
    valid = da.notnull() & xr.apply_ufunc(np.isfinite, da, dask="allowed")
    if fill_val is not None:
        valid = valid & (da != fill_val)
    else:
        for fv in _COMMON_FILL_VALUES:
            valid = valid & (abs(da - fv) > 1e-3)
    return xr.where(valid, da + K0, da)


# ---------------------------------------------------------------------------
# Helpers (unchanged from palm_output_coordinate_pospr.py)
# ---------------------------------------------------------------------------

def _is_valid_time(time_values, threshold=1e10):
    """Return a boolean mask of timesteps with plausible values."""
    if np.issubdtype(time_values.dtype, np.datetime64):
        return np.isfinite(time_values)
    try:
        if np.any(np.abs(time_values) >= threshold):
            return np.isfinite(time_values)
        return np.abs(time_values) < threshold
    except (TypeError, np.core._exceptions._UFuncNoLoopError):
        return np.isfinite(time_values)


def _normalize_time_units(units_str: str) -> str:
    """
    Normalise a CF time units string so that QGIS/MDAL can parse it.
    Converts sub-second units to 'seconds since <epoch>', strips ISO T
    separator and timezone suffixes.
    """
    m = re.match(
        r"(nanoseconds|microseconds|milliseconds)\s+since\s+(.+)",
        units_str.strip(), re.IGNORECASE,
    )
    if m:
        epoch_str = m.group(2).strip()
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
            try:
                epoch_dt = datetime.strptime(epoch_str, fmt)
                break
            except ValueError:
                continue
        else:
            units_str = re.sub(m.group(1), "seconds", units_str, flags=re.IGNORECASE)
            epoch_dt = None

        if epoch_dt is not None:
            epoch_whole = epoch_dt.replace(microsecond=0)
            return f"seconds since {epoch_whole.strftime('%Y-%m-%d %H:%M:%S')}"

    u = re.sub(r"(\d)T(\d)", r"\1 \2", units_str)
    u = re.sub(r"\s*[+-]\d{2}(:?\d{2})?\s*$", "", u)
    u = re.sub(r"\s*UTC\s*$", "", u).strip()
    return u


# ---------------------------------------------------------------------------
# WGS84 auxiliary coordinates
# ---------------------------------------------------------------------------

def _add_latlon(
    ds_out:     xr.Dataset,
    target_crs: "pyproj.CRS",
    log:        logging.Logger,
) -> xr.Dataset:
    """
    Add CF-compliant 2D auxiliary latitude/longitude coordinates.

    A PALM domain is a regular grid in its PROJECTED CRS, so it is not
    regular in lon/lat — the geographic coordinates therefore cannot be
    1-D dimension coordinates. CF handles exactly this case with 2-D
    auxiliary coordinate variables on (y, x), referenced from each data
    variable's `coordinates` attribute. QGIS, ArcGIS, xarray and Panoply
    all understand this form.

    The projected dimension coordinates are left untouched.
    """
    if "x" not in ds_out.coords or "y" not in ds_out.coords:
        log.warning("[coord] no x/y coordinates - cannot build lat/lon.")
        return ds_out

    x_vals = np.asarray(ds_out["x"].values, dtype=np.float64)
    y_vals = np.asarray(ds_out["y"].values, dtype=np.float64)
    x2d, y2d = np.meshgrid(x_vals, y_vals)          # shape (y, x)

    transformer = pyproj.Transformer.from_crs(
        target_crs, pyproj.CRS.from_epsg(4326), always_xy=True,
    )
    lon2d, lat2d = transformer.transform(x2d, y2d)

    n_bad = int(np.sum(~np.isfinite(lon2d) | ~np.isfinite(lat2d)))
    if n_bad:
        log.warning("[coord] %d point(s) outside the projection's validity "
                    "- no WGS84 coordinates there.", n_bad)

    ds_out["latitude"] = xr.Variable(
        ("y", "x"), lat2d.astype(np.float64),
        attrs={"units": "degrees_north", "standard_name": "latitude",
               "long_name": "latitude (WGS84)"},
    )
    ds_out["longitude"] = xr.Variable(
        ("y", "x"), lon2d.astype(np.float64),
        attrs={"units": "degrees_east", "standard_name": "longitude",
               "long_name": "longitude (WGS84)"},
    )

    log.debug("[coord] WGS84 grid: lat %.5f..%.5f, lon %.5f..%.5f",
              float(np.nanmin(lat2d)), float(np.nanmax(lat2d)),
              float(np.nanmin(lon2d)), float(np.nanmax(lon2d)))
    return ds_out


def _parse_cf_epoch(units_str: str):
    """
    Return the epoch of a CF time-units string as a UTC np.datetime64, or
    None if it cannot be parsed.

    Any trailing UTC offset ("+02:00", "-0500") is APPLIED, not stripped —
    xarray's decode_cf honours it, so the numeric fallback path must too or
    the two paths would disagree by the offset.
    """
    m = re.search(
        r"since\s+(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?)?)"
        r"\s*(Z|UTC|[+-]\d{2}:?\d{2})?",
        units_str.strip(), re.IGNORECASE,
    )
    if not m:
        return None

    stamp = m.group(1).replace("T", " ")
    if "." in stamp:
        stamp = stamp.split(".")[0]
    if len(stamp) == 10:                      # date only
        stamp += " 00:00:00"
    try:
        epoch = np.datetime64(stamp, "s")
    except ValueError:
        return None

    tz = (m.group(2) or "").strip()
    if tz and tz.upper() not in ("Z", "UTC"):
        sign = 1 if tz[0] == "+" else -1
        digits = tz[1:].replace(":", "")
        offset_s = sign * (int(digits[:2]) * 3600 + int(digits[2:4]) * 60)
        # "09:00+02:00" is 07:00 UTC — subtract the offset to reach UTC.
        epoch = epoch - np.timedelta64(offset_s, "s")
    return epoch


def _is_netcdf(path: Path) -> bool:
    """Detect NetCDF by magic bytes."""
    MAGIC = (b"CDF\x01", b"CDF\x02", b"CDF\x05", b"\x89HDF\r\n\x1a\n")
    try:
        with open(path, "rb") as fh:
            header = fh.read(8)
        return any(header.startswith(m) for m in MAGIC)
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Core coordinate logic (lifted from process_file() in the original script)
# ---------------------------------------------------------------------------

def _add_coordinates(
    ds:         xr.Dataset,
    coord_cfg:  CoordConfig,
    src_path:   Path,
    log:        logging.Logger,
) -> xr.Dataset:
    """
    Add UTM coordinates and CRS metadata to *ds*.
    Returns the augmented dataset (ready to write).

    Parameters
    ----------
    ds          : open dataset (decode_times=False, decode_cf=False)
    coord_cfg   : CoordConfig (.crs, .utm_zone)
    src_path    : original file path (used only for error messages)
    log         : logger
    """
    original_units = ds["time"].attrs.get("units", "seconds since 1970-01-01")

    # --- Filter time (keep as numeric) ------------------------------------
    log.debug("[coord] filtering time")
    valid_indices = np.where(_is_valid_time(ds.time.values))[0]
    if len(valid_indices) == 0:
        raise ValueError(f"No valid timesteps found in {src_path.name}.")
    ds = ds.isel(time=valid_indices)

    time_attrs = ds["time"].attrs.copy()

    ds_decoded = xr.decode_cf(ds)
    time_datetime = ds_decoded["time"].values

    origin_dt64 = np.datetime64("1970-01-01", "s")

    _origin_time_attr = ds.attrs.get("origin_time", "")
    if _origin_time_attr:
        _ot = re.sub(r"\s*[+-]\d{2}(:?\d{2})?\s*$", "", _origin_time_attr.strip())
        _ot = re.sub(r"\s*UTC\s*$", "", _ot, flags=re.IGNORECASE)
        _ot = re.sub(r"\s*Z\s*$",   "", _ot).strip()
        origin_time_units = f"seconds since {_ot}"
        origin_dt64 = np.datetime64(_ot, "s")
    else:
        origin_time_units = _normalize_time_units(original_units)
        m = re.search(r"since\s+(\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2}:\d{2})?)", origin_time_units)
        if m:
            origin_dt64 = np.datetime64(m.group(1), "s")

    # Read averaging interval from global attributes. Absent means this is
    # NOT an averaged product (instantaneous output), in which case the
    # timestamps must NOT be snapped to a grid — see below.
    _dt_attr = ds.attrs.get("dt_averaging", ds.attrs.get("averaging_interval"))
    dt = float(_dt_attr) if _dt_attr is not None else None
    if dt is not None and dt <= 0:
        log.warning("[coord] %s: dt_averaging=%.1f is invalid - timestamps "
                    "not snapped.", src_path.name, dt)
        dt = None
    log.debug("[coord] averaging interval dt=%s",
              f"{dt:.1f} s" if dt else "none (instantaneous output)")

    # time_datetime may come back either as datetime64 (if decode_cf parsed
    # the CF units) or as plain numeric seconds (if it could not). Handle
    # both so we never mix float64 and datetime64 in one subtraction.
    if np.issubdtype(np.asarray(time_datetime).dtype, np.datetime64):
        elapsed_seconds = (time_datetime - origin_dt64) / np.timedelta64(1, "s")
    else:
        # Numeric seconds since the FILE's own epoch, which is not
        # necessarily origin_time — a restarted run can carry a different
        # CF epoch. Parse the file's epoch and shift by the difference
        # rather than assuming the two coincide.
        elapsed_seconds = np.asarray(time_datetime, dtype=np.float64)
        file_epoch = _parse_cf_epoch(original_units)
        if file_epoch is not None:
            shift = (file_epoch - origin_dt64) / np.timedelta64(1, "s")
            if shift != 0.0:
                log.debug("[coord] file epoch differs from origin_time by "
                          "%.0f s - shifting time values.", shift)
                elapsed_seconds = elapsed_seconds + shift
        else:
            log.warning("[coord] %s: no CF epoch in time units '%s' - taken "
                        "as seconds since origin_time.", src_path.name,
                        original_units)

    # Snap to the averaging interval only for averaged products. PALM writes
    # averaged output at exact multiples of dt_averaging, so rounding removes
    # float noise there — but applying it to instantaneous output would move
    # genuine timestamps onto a grid they were never on.
    if dt is not None:
        time_seconds = np.round(elapsed_seconds / dt) * dt
    else:
        time_seconds = elapsed_seconds

    ds = ds.assign_coords(time=("time", time_seconds.astype(np.float64)))
    ds["time"].encoding.clear()

    # --- Determine EPSG ---------------------------------------------------
    log.debug("[coord] resolving CRS")
    _epsg = coord_cfg.utm_zone  # None unless user overrides in config
    if _epsg is None:
        if "crs" in ds and "epsg_code" in ds["crs"].attrs:
            epsg_str = ds["crs"].attrs.get("epsg_code", "")
            try:
                _epsg = int(epsg_str.replace("EPSG:", "").strip())
            except ValueError:
                pass
    if _epsg is None:
        _epsg = 32633
        # A domain outside UTM 33N would get the wrong CRS label.
        log.warning("[coord] %s: no EPSG in the file or config - assuming "
                    "EPSG:%d (UTM 33N); set steps.coord.utm_zone.",
                    src_path.name, _epsg)

    target_crs = pyproj.CRS.from_epsg(_epsg)

    # --- Read domain origin -----------------------------------------------
    if "origin_x" not in ds.attrs or "origin_y" not in ds.attrs:
        raise KeyError(
            f"{src_path.name}: missing global attributes 'origin_x' and/or 'origin_y'."
        )
    origin_x = ds.attrs["origin_x"]
    origin_y = ds.attrs["origin_y"]
    origin_z = ds.attrs.get("origin_z", 0.0)

    # --- Build UTM coordinates --------------------------------------------
    log.debug("[coord] building UTM coordinates")
    ds_out = ds

    GRID_DEFS = [
        ("x",  "y",  "E_UTM",  "N_UTM"),
        ("xu", "y",  "Eu_UTM", "Nu_UTM"),
        ("x",  "yv", "Ev_UTM", "Nv_UTM"),
    ]

    for dim, offset, axis, std_name in [
        ("x",     origin_x, "X", "projection_x_coordinate"),
        ("xu",    origin_x, "X", "projection_x_coordinate"),
        ("y",     origin_y, "Y", "projection_y_coordinate"),
        ("yv",    origin_y, "Y", "projection_y_coordinate"),
        ("z",     origin_z, "Z", "height"),
        ("zu_3d", origin_z, "Z", "height"),
        ("zw_3d", origin_z, "Z", "height"),
    ]:
        if dim not in ds_out.coords:
            continue
        utm_vals = ds_out[dim].values + offset
        ds_out[dim] = xr.Variable(
            dim, utm_vals,
            attrs={
                "units":         "m",
                "standard_name": std_name,
                "long_name":     (
                    f"UTM {'easting' if axis == 'X' else ('northing' if axis == 'Y' else 'height')}"
                    f" ({dim})"
                ),
                "axis": axis,
            },
        )

    _crs_mode = str(coord_cfg.crs).strip().lower()
    _want_utm = _crs_mode in ("utm", "both")
    _want_wgs = _crs_mode in ("wgs84", "both")

    if _want_utm:
        for x_dim, y_dim, e_var, n_var in GRID_DEFS:
            if x_dim in ds_out.coords:
                ds_out[e_var] = xr.Variable(
                    x_dim, ds_out[x_dim].values,
                    attrs={"units": "m", "standard_name": "projection_x_coordinate",
                           "long_name": "UTM easting"},
                )
            if y_dim in ds_out.coords:
                ds_out[n_var] = xr.Variable(
                    y_dim, ds_out[y_dim].values,
                    attrs={"units": "m", "standard_name": "projection_y_coordinate",
                           "long_name": "UTM northing"},
                )

    if _want_wgs:
        log.debug("[coord] building WGS84 auxiliary coordinates")
        ds_out = _add_latlon(ds_out, target_crs, log)

    # --- Put temperature variables on the requested scale ------------------
    # `celsius` selects the OUTPUT scale; the source scale is read from
    # each variable's `units`. Conversion happens only when the two
    # differ, so an already-Celsius PALM variable (ta, ta_2m*) is left
    # alone instead of being driven 273.15 below reality. `theta` is
    # exempt entirely — see CELSIUS_EXEMPT_PREFIXES.
    target = "C" if coord_cfg.celsius else "K"
    if not coord_cfg.celsius:
        log.debug("[coord] celsius disabled - temperatures emitted in kelvin")

    overrides = dict(getattr(coord_cfg, "temperature_units", None) or {})

    for var in list(ds_out.data_vars):
        if not is_temperature(var):
            continue
        if is_celsius_exempt(var):
            log.debug("[coord] %s: potential temperature, left in kelvin", var)
            continue

        da = ds_out[var]
        fill_val = da.attrs.get("_FillValue", da.encoding.get("_FillValue"))
        src_units = da.attrs.get("units", da.encoding.get("units"))
        source = temperature_scale(src_units, var, overrides)

        if source == target:
            log.debug("[coord] %s: already in %s (units=%r) - not converted",
                      var, target, src_units)
            ds_out[var].attrs["palm_postproc_source_units"] = str(src_units or "")
            ds_out[var].attrs["palm_postproc_output_units"] = str(src_units or "")
            continue

        # Lazy: xr.where keeps dask arrays as dask arrays, so a chunked 3D
        # file is converted chunk by chunk at write time instead of being
        # pulled into memory whole (the previous da.values path materialised
        # the full array twice per variable, defeating the chunking above).
        if source == "K":
            converted = kelvin_to_celsius_da(da, fill_val)
            out_units = "degrees_C"
        else:
            converted = celsius_to_kelvin_da(da, fill_val)
            out_units = "K"

        ds_out[var] = converted
        ds_out[var].attrs = dict(da.attrs)
        ds_out[var].attrs["units"] = out_units
        ds_out[var].attrs["palm_postproc_source_units"] = str(src_units or "")
        ds_out[var].attrs["palm_postproc_output_units"] = out_units
        if fill_val is not None:
            ds_out[var].attrs["_FillValue"] = fill_val
        # Fill / invalid points are left untouched.
        log.info("[coord] %s: converted %s -> %s", var, source, out_units)

    # --- Write CRS metadata -----------------------------------------------
    log.debug("[coord] writing CRS metadata")
    ds_out.attrs["Conventions"] = "CF-1.7"

    crs_cf = target_crs.to_cf()
    ds_out["crs"] = xr.Variable(
        (), np.int32(_epsg),
        attrs={
            "epsg_code":                        f"EPSG:{_epsg}",
            "grid_mapping_name":                crs_cf.get("grid_mapping_name", "transverse_mercator"),
            "longitude_of_central_meridian":    crs_cf.get("longitude_of_central_meridian", ""),
            "false_easting":                    crs_cf.get("false_easting", 500000.0),
            "false_northing":                   crs_cf.get("false_northing", 0.0),
            "scale_factor_at_central_meridian": crs_cf.get("scale_factor_at_central_meridian", 0.9996),
            "long_name":                        "coordinate reference system",
            "crs_wkt":                          target_crs.to_wkt(),
        },
    )

    for var in list(ds_out.data_vars):
        if var != "crs":
            ds_out[var].attrs["grid_mapping"] = "crs"

    # CF: data variables must point at their auxiliary coordinates, or
    # readers will not associate the 2-D lat/lon grid with the data.
    if _want_wgs and "latitude" in ds_out and "longitude" in ds_out:
        for var in list(ds_out.data_vars):
            if var in ("crs", "latitude", "longitude"):
                continue
            if {"x", "y"}.issubset(set(ds_out[var].dims)):
                existing = ds_out[var].attrs.get("coordinates", "")
                parts = [p for p in existing.split() if p not in
                         ("latitude", "longitude")]
                ds_out[var].attrs["coordinates"] = " ".join(
                    parts + ["latitude", "longitude"])

    ds_out["time"].attrs["units"]    = origin_time_units
    ds_out["time"].attrs["calendar"] = "standard"

    return ds_out


# ---------------------------------------------------------------------------
# Per-file processor
# ---------------------------------------------------------------------------

def _output_name(src_path: Path) -> str:
    """Derive output filename: insert .utm before the final .nc suffix."""
    stem = src_path.stem
    return f"{stem}{_TRANSFORMED_SUFFIX}.nc"


def _process_file(
    src_path:  Path,
    out_dir:   Path,
    cfg:       Config,
    dry_run:   bool,
    log:       logging.Logger,
) -> bool:
    """Process one file. Returns True on success."""
    if not _is_netcdf(src_path):
        log.warning("[coord] %s is not a valid NetCDF file - skipped.",
                    src_path.name)
        return True

    is_3d  = "_av_3d" in src_path.stem
    chunks = try_dask_chunks(_CHUNKS_2D, _CHUNKS_3D, is_3d)

    out_path = out_dir / _output_name(src_path)
    if not should_write(out_path, cfg.overwrite, log, "coord"):
        return True

    if dry_run:
        log.info("[coord] would write %s: %s coordinates (dry run)",
                 out_path.name, cfg.steps.coord.crs)
        return True

    ds = xr.open_dataset(src_path, decode_times=False, decode_cf=False, chunks=chunks)

    # Skip time-invariant files (e.g. ind_z_xy, zusi, zwwi) — they carry
    # no time dimension and do not need coordinate transformation.
    if "time" not in ds.dims and "time" not in ds.coords:
        log.debug("[coord] %s is time-invariant - skipped", src_path.name)
        ds.close()
        return True

    log.debug("[coord] adding coordinates: %s -> %s", src_path.name,
              out_path.name)
    t0 = time.monotonic()

    try:
        ds_out = _add_coordinates(ds, cfg.steps.coord, src_path, log)
        encoding = {
            var: {"zlib": True, "complevel": cfg.complevel, "shuffle": True}
            for var in ds_out.data_vars
        }
        encoding["time"] = {"dtype": "float64"}
        ds_out.to_netcdf(out_path, format="NETCDF4", encoding=encoding)
        log.info("[coord] wrote %s: %s in %s", out_path.name,
                 fmt_size(out_path), fmt_elapsed(t0))
        return True
    except Exception as exc:
        log.error("[coord] %s failed: %s", out_path.name, exc)
        log.debug("[coord] traceback:", exc_info=True)
        return False
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(cfg: Config, dry_run: bool, log: logging.Logger) -> None:
    """
    Run the coord step for all .nc files in cfg.paths.output_splittime.
    Raises RuntimeError if any file fails.
    """
    step(log, "Georeferencing the coordinates")

    input_dir = cfg.paths.output_splittime
    out_dir   = cfg.paths.output_coord

    nc_files = sorted(input_dir.glob("*.nc"))
    if not nc_files:
        log.warning("[coord] no .nc files found in %s", input_dir)
        return

    log.info("[coord] %d file(s) to process", len(nc_files))

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for src_path in nc_files:
        if not _process_file(src_path, out_dir, cfg, dry_run, log):
            failures += 1

    if failures:
        raise RuntimeError(f"[coord] {failures} file(s) failed - see the log above.")

