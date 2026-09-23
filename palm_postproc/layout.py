"""
palm_postproc.layout
--------------------
The newcomer config layout, translated onto the internal settings.

A config in this layout says what to READ (`input`), what to keep
(`variables`, `region`, `time`) and where the two outputs go (`joined`,
`analysis`). Rarely needed settings live in `advanced`. See template.yaml.

Everything downstream still works on the internal structure (`case`,
`paths`, `steps`, ...), which is also what the pre-0.6 layout writes
directly. This module only maps one onto the other, and checks the new
layout strictly: an unknown key is an error with a suggestion, because a
silently ignored typo is the commonest way a config does not do what it
says.

The same section names and the same vocabulary as palm2gis and
palm_preproc: `project`, `input`, `region`, `time`, `advanced`.

Public API
----------
  is_new_layout(raw)  ->  bool
  translate(raw)      ->  (legacy-style dict, notes)
"""

from __future__ import annotations

import difflib

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = {
    "project": {"name": None, "root": None, "output_dir": None,
                "overwrite": None},
    "input": {"palm_output": None, "domain": None, "joined": None,
              "crs": None},
    "variables": {"av_xy": None, "av_3d": None, "group": None,
                  "group_suffix": None},
    "region": {"z_max": None},
    "time": {"from": None, "to": None, "step": None},
    "joined": {"dir": None},
    "analysis": {"dir": None, "coordinates": None},
    "advanced": {
        "units": {"temperature": None, "potential_temperature": None,
                  "overrides": None},
        "performance": {"complevel": None, "workers": None,
                        "keep_intermediate": None},
        "z_coord": None,
        "join": {"files": None, "naming": None, "file_prefix": None,
                 "file_suffix": None, "complevel": None, "merge_tol": None,
                 "create_new_file": None, "offset_min": None,
                 "part_timeshift": None, "output_timestep": None},
    },
}

# Top-level keys of the pre-0.6 layout, and where each went.
_LEGACY_HINT = {
    "case": "project.name",
    "paths": "project.root, input.palm_output, joined.dir and analysis.dir",
    "steps": "variables, region, time, analysis and advanced",
    "chain": "advanced.performance.keep_intermediate (inverted)",
    "complevel": "advanced.performance.complevel",
    "workers": "advanced.performance.workers",
    "domain": "input.domain",
    "verbosity": "the -v / -q command-line flags",
}


class LayoutError(Exception):
    pass


# ---------------------------------------------------------------------------
# Detection and key checking
# ---------------------------------------------------------------------------

def is_new_layout(raw):
    """The new layout is recognised by its own top-level blocks."""
    return isinstance(raw, dict) and any(
        k in raw for k in ("project", "input", "variables", "joined",
                           "analysis", "region", "time", "advanced"))


def _check_keys(block, schema, where):
    if block is None:
        return
    if not isinstance(block, dict):
        raise LayoutError(f"{where or 'the config'} must be a block of "
                          f"settings, got {block!r}.")
    for key, val in block.items():
        path = f"{where}.{key}" if where else key
        if key not in schema:
            if not where and key in _LEGACY_HINT:
                raise LayoutError(
                    f"'{key}' belongs to the old config layout. In this "
                    f"layout it is {_LEGACY_HINT[key]}.")
            close = difflib.get_close_matches(key, list(schema), n=1)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            raise LayoutError(f"Unknown setting '{path}'.{hint} Known "
                              f"here: {', '.join(schema)}.")
        if isinstance(schema[key], dict):
            _check_keys(val, schema[key], path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bool(val, where):
    """true/false, or None when unset. YAML reads "false" in quotes as a
    non-empty string, which Python would take as True."""
    if val is None or isinstance(val, bool):
        return val
    raise LayoutError(f"{where} must be true or false (without quotes), "
                      f"got {val!r}.")


def _dir(block, where):
    """The `dir:` line of an output block, or None when the block is
    missing or its dir is null."""
    if block is None:
        return None
    if not isinstance(block, dict):
        raise LayoutError(f"{where} must be a block with a dir: line, got "
                          f"{block!r}.")
    d = block.get("dir")
    if d is None:
        return None
    if not isinstance(d, str) or not d.strip():
        raise LayoutError(f"{where}.dir must be a directory or null, got "
                          f"{d!r}.")
    return d.strip()


def _under(root, path, name):
    """A path from the config, expanded and anchored at project.root."""
    text = str(path).replace("{name}", name)
    if text.startswith(("/", "~")):
        return text
    return f"{root.rstrip('/')}/{text.lstrip('./')}"


def _units(adv, notes):
    """advanced.units -> the internal steps.coord.celsius flag."""
    units = adv.get("units") or {}
    pot = units.get("potential_temperature")
    if pot is not None and str(pot).upper() != "K":
        raise LayoutError(
            "advanced.units.potential_temperature must be K: theta* is "
            "kelvin-defined, so a Celsius value would not be a potential "
            "temperature.")
    scale = units.get("temperature")
    if scale is None:
        return None
    if str(scale).upper() not in ("K", "C"):
        raise LayoutError(f"advanced.units.temperature must be K or C, got "
                          f"{scale!r}.")
    return str(scale).upper() == "C"


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def translate(raw):
    """New-layout dict -> the internal settings dict (still merged with
    the defaults afterwards). Also returns notes for the log."""
    _check_keys(raw, SCHEMA, "")
    notes = []
    out = {}

    project = raw.get("project") or {}
    if not project.get("name"):
        raise LayoutError("project.name is required (the PALM job name).")
    name = str(project["name"])
    if not project.get("root"):
        raise LayoutError("project.root is required (the PALM job "
                          "directory).")
    root = str(project["root"]).replace("{name}", name)
    out["case"] = name
    out["overwrite"] = _bool(project.get("overwrite"), "project.overwrite")
    if out["overwrite"] is None:
        del out["overwrite"]

    inp = raw.get("input") or {}
    if not isinstance(inp, dict):
        raise LayoutError(f"input must be a block of settings, got {inp!r}.")
    if inp.get("domain") is not None:
        out["domain"] = inp["domain"]
    if inp.get("crs") is not None:
        out.setdefault("steps", {}).setdefault("coord", {})["utm_zone"] = \
            _epsg(inp["crs"])

    adv = raw.get("advanced") or {}
    if not isinstance(adv, dict):
        raise LayoutError(f"advanced must be a block of settings, got "
                          f"{adv!r}.")
    perf = adv.get("performance") or {}

    # -- paths ---------------------------------------------------------------
    out_dir = _under(root, project.get("output_dir") or ".", name)
    joined_dir = _dir(raw.get("joined"), "joined")
    analysis_dir = _dir(raw.get("analysis"), "analysis")
    paths = {"base": root}
    paths["join_input"] = _under(root, inp.get("palm_output") or "OUTPUT",
                                 name)
    if inp.get("joined"):
        paths["input"] = _under(root, inp["joined"], name)
    if joined_dir:
        paths["output_join"] = _under(out_dir, joined_dir, name)
    keep = _bool(perf.get("keep_intermediate"),
                 "advanced.performance.keep_intermediate")
    if analysis_dir:
        final = _under(out_dir, analysis_dir, name)
        for key, suffix in (("output_splitvar", "_splitvar"),
                            ("output_splitz", "_splitz"),
                            ("output_splittime", "_splittime"),
                            ("output_coord", "")):
            # Chained in memory, only the last enabled step writes, so every
            # post-join step points at the analysis directory. Written out
            # step by step, each needs its own.
            paths[key] = f"{final}{suffix}" if keep and suffix else final
    out["paths"] = paths

    # -- what runs -----------------------------------------------------------
    steps = out.setdefault("steps", {})
    steps.setdefault("join", {})["enabled"] = bool(
        joined_dir and not inp.get("joined"))
    if inp.get("joined") and joined_dir:
        notes.append("input.joined is set, so the files are read from there "
                     "and joined.dir is not written.")
    analysis_on = bool(analysis_dir)
    steps.setdefault("splitvar", {})["enabled"] = analysis_on

    region = raw.get("region") or {}
    if not isinstance(region, dict):
        raise LayoutError(f"region must be a block of settings, got "
                          f"{region!r}.")
    z_max = region.get("z_max")
    steps.setdefault("splitz", {})["enabled"] = analysis_on and z_max is not None
    if z_max is not None:
        if not isinstance(z_max, (int, float)) or isinstance(z_max, bool):
            raise LayoutError(f"region.z_max must be a number of metres, "
                              f"got {z_max!r}.")
        steps["splitz"]["z_max"] = float(z_max)
    if adv.get("z_coord") is not None:
        steps.setdefault("splitz", {})["z_coord"] = adv["z_coord"]

    window = raw.get("time") or {}
    if not isinstance(window, dict):
        raise LayoutError(f"time must be a block with from:, to: and step: "
                          f"lines, got {window!r}.")
    t_from, t_to, t_step = (window.get("from"), window.get("to"),
                            window.get("step"))
    steps.setdefault("splittime", {})["enabled"] = analysis_on and any(
        v is not None for v in (t_from, t_to, t_step))
    steps["splittime"]["time_from"] = t_from
    steps["splittime"]["time_to"] = t_to
    if t_step is not None:
        steps["splittime"]["timestep"] = str(t_step)

    coords = (raw.get("analysis") or {}).get("coordinates", "utm")
    steps.setdefault("coord", {})["enabled"] = analysis_on and bool(coords)
    if coords:
        steps["coord"]["crs"] = coords

    # -- variables -----------------------------------------------------------
    var = raw.get("variables") or {}
    if not isinstance(var, dict):
        raise LayoutError(f"variables must be a block of settings, got "
                          f"{var!r}.")
    for key, internal in (("av_xy", "vars_2d"), ("av_3d", "vars_3d"),
                          ("group", "group_vars"),
                          ("group_suffix", "group_suffix")):
        if var.get(key) is not None:
            steps.setdefault("splitvar", {})[internal] = var[key]

    # -- advanced ------------------------------------------------------------
    celsius = _units(adv, notes)
    if celsius is not None:
        steps.setdefault("coord", {})["celsius"] = celsius
    if (adv.get("units") or {}).get("overrides") is not None:
        steps.setdefault("coord", {})["temperature_units"] = \
            adv["units"]["overrides"]
    for key, internal in (("complevel", "complevel"), ("workers", "workers")):
        if perf.get(key) is not None:
            out[internal] = perf[key]
    if keep is not None:
        out["chain"] = not keep

    join = adv.get("join") or {}
    if not isinstance(join, dict):
        raise LayoutError(f"advanced.join must be a block of settings, got "
                          f"{join!r}.")
    for key, internal in (("files", "filelist"), ("naming", "convention"),
                          ("file_prefix", "file_prefix"),
                          ("file_suffix", "file_suffix"),
                          ("complevel", "complevel"),
                          ("merge_tol", "merge_tol"),
                          ("create_new_file", "create_new_file"),
                          ("offset_min", "offset_min"),
                          ("part_timeshift", "part_timeshift"),
                          ("output_timestep", "output_timestep")):
        if join.get(key) is not None:
            steps.setdefault("join", {})[internal] = join[key]
    if isinstance(steps["join"].get("filelist"), list):
        steps["join"]["filelist"] = [str(f).replace("{name}", name)
                                     for f in steps["join"]["filelist"]]

    return out, notes


def _epsg(crs):
    """input.crs ('EPSG:32633' or a bare code) -> the internal utm_zone."""
    text = str(crs).strip().upper()
    if text.startswith("EPSG:"):
        text = text[5:]
    if not text.isdigit():
        raise LayoutError(f"input.crs must be an EPSG code, e.g. "
                          f"'EPSG:32633'; got {crs!r}.")
    return int(text)


# ---------------------------------------------------------------------------
# Messages in the user's vocabulary
# ---------------------------------------------------------------------------
# Step code speaks the internal names. For a config in the new layout, the
# log rewrites them into the names that config uses.

import re

MESSAGE_NAMES = {
    "steps.splitz.z_max": "region.z_max",
    "steps.splitz.z_coord": "advanced.z_coord",
    "steps.splittime.timestep": "time.step",
    "steps.splitvar.vars_2d": "variables.av_xy",
    "steps.splitvar.vars_3d": "variables.av_3d",
    "steps.coord.celsius": "advanced.units.temperature",
    "steps.coord.utm_zone": "input.crs",
    "steps.coord.crs": "analysis.coordinates",
    "steps.join.convention": "advanced.join.naming",
    "steps.join.filelist": "advanced.join.files",
    "steps.join.merge_tol": "advanced.join.merge_tol",
    "paths.join_input": "input.palm_output",
    "paths.output_join": "joined.dir",
    "paths.output_coord": "analysis.dir",
    "complevel": "advanced.performance.complevel",
    "workers": "advanced.performance.workers",
}

_MESSAGE_RE = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(k) for k in
                           sorted(MESSAGE_NAMES, key=len, reverse=True))
    + r")(?!\w)")


def to_new_names(text):
    """Rewrite internal setting names in a message into the new layout."""
    return _MESSAGE_RE.sub(lambda m: MESSAGE_NAMES[m.group(1)], text)
