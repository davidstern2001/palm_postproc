"""
palm_postproc.config
--------------------
Loads, validates, and resolves a palm_postproc YAML configuration file.

Public API
----------
  load(path)  →  Config
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import yaml

from .layout import LayoutError, is_new_layout, translate

log = logging.getLogger("palm_postproc")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULTS: dict = {
    "domain":    None,
    "complevel": 4,
    "overwrite": False,
    "verbosity": "info",
    "chain":     True,
    "workers":   1,
    "steps": {
        "join": {
            "enabled":        True,
            "filelist":       "all",
            "convention":     "filenum",
            "file_prefix":    "",
            "file_suffix":    ".nc",
            "create_new_file": True,
            "offset_min":     0,
            "part_timeshift": 0,
            "output_timestep": 0,
            "merge_tol":      None,
            "complevel":      6,
        },
        "splitvar": {
            "enabled":      True,
            "vars_2d":      "all",
            "vars_3d":      "all",
            "group_vars":   ["u", "v", "w"],
            "group_suffix": "uvw",
        },
        "splitz": {
            "enabled": True,
            "z_max":   300.0,
            "z_coord": "zu_3d",
        },
        "splittime": {
            "enabled":  True,
            # Window in time (the `time:` section): both ends inclusive,
            # resolved against each file's own axis. See timespec.py.
            "time_from": None,
            "time_to":   None,
            # Window by record index (pre-0.6, and the escape hatch when a
            # file carries no usable times).
            "t_start":  None,
            "t_end":    None,
            "timestep": None,
        },
        "coord": {
            "enabled":  True,
            "crs":      "utm",
            "utm_zone": None,
            # Set by advanced.units.temperature (C | K), and C by
            # default since 0.6.0: an air or surface temperature is wanted
            # in degrees C. Covers tsurf*, t_surf* and ta*. A POTENTIAL
            # temperature (theta*) is kelvin-defined and is exempt whatever
            # this says. The same rule and the same variable list as
            # palm2gis, so a directory cannot end up mixing units.
            "celsius":  True,
            # Explicit temperature scale for variables whose `units`
            # attribute cannot be read. Maps a variable name to "K" or
            # "C", e.g. {ta_2m*_xy: C}. Needed rarely: PALM's own unit
            # strings are recognised, including the truncated "degree_"
            # it writes from a fixed-length buffer. The step RAISES on an
            # unrecognised unit rather than guessing, so this is the
            # escape hatch when a new PALM version writes something new.
            "temperature_units": {},
        },
    },
}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PathsConfig:
    base:             Path
    join_input:       Path            # raw PALM OUTPUT dir (input to join step)
    output_join:      Path            # joined files (input to post-join steps)
    output_splitvar:  Path
    output_splitz:    Path
    output_splittime: Path
    output_coord:     Path
    input_override:   Optional[Path]  # explicit override: skip join, read from here


@dataclass
class JoinConfig:
    enabled:         bool
    filelist:        Union[str, List[str]]   # "all" or explicit list
    convention:      str
    file_prefix:     str
    file_suffix:     str
    create_new_file: bool
    offset_min:      int
    part_timeshift:  int
    output_timestep: int
    merge_tol:       Optional[float]
    complevel:       int


@dataclass
class SplitvarsConfig:
    enabled:      bool
    vars_2d:      Union[str, List[str]]
    vars_3d:      Union[str, List[str]]
    group_vars:   List[str]
    group_suffix: str


@dataclass
class SplitzConfig:
    enabled: bool
    z_max:   float
    z_coord: str


@dataclass
class SplittimeConfig:
    enabled:   bool
    time_from: Optional[object]
    time_to:   Optional[object]
    t_start:   Optional[int]
    t_end:     Optional[int]
    timestep:  Optional[str]


@dataclass
class CoordConfig:
    enabled:  bool
    crs:      str
    utm_zone: Optional[int]
    celsius:  bool
    temperature_units: dict


@dataclass
class StepsConfig:
    join:      JoinConfig
    splitvar:  SplitvarsConfig
    splitz:    SplitzConfig
    splittime: SplittimeConfig
    coord:     CoordConfig


@dataclass
class Config:
    case:      str
    domain:    Optional[str]
    complevel: int
    overwrite: bool
    verbosity: str
    chain:     bool
    workers:   int
    paths:     PathsConfig
    steps:     StepsConfig
    layout:    str = "legacy"       # "new" (template.yaml) or "legacy"
    _source:   Path = field(repr=False, default=None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _parse_domain_suffix(domain) -> str:
    """
    Normalise the 'domain' config value to a PALM-style suffix.

    Accepts, all producing '_N02':
      2          (int)
      "2"        (bare digit string)
      "02"       (zero-padded string)
      "N02"      (with N prefix)
      "_N02"     (full suffix form)

    None, 0, "", or "0" all mean root domain → returns "".
    """
    if domain is None:
        return ""
    if isinstance(domain, int):
        return f"_N{domain:02d}" if domain > 0 else ""

    s = str(domain).strip()
    if s == "" or s == "0":
        return ""

    # Already a full suffix like "_N02"
    if re.fullmatch(r"_N\d+", s):
        return s

    # "N02" → strip leading N
    if re.fullmatch(r"N\d+", s, re.IGNORECASE):
        s = s[1:]

    # Now s should be a plain digit string, e.g. "2" or "02"
    if not s.isdigit():
        raise ValueError(
            f"Invalid 'domain' value: {domain!r}. "
            "Use an integer (2), or a string like '2', '02', 'N02', or '_N02'."
        )
    return f"_N{int(s):02d}"


def _resolve_path_template(template: str, case: str, domain_suffix: str,
                            base: str, outbase: str) -> Path:
    s = (template
         .replace("{case}", case)
         .replace("{domain}", domain_suffix)
         .replace("{outbase}", outbase)
         .replace("{base}", base))
    return Path(s)


def _build_paths(raw_paths: dict, case: str, domain_suffix: str) -> PathsConfig:
    if "base" not in raw_paths:
        raise ValueError("paths.base is required in the configuration file.")

    base    = Path(raw_paths["base"]).expanduser()
    # outbase = {base}/{case} — the individual case directory.
    # base is expected to be the JOBS root (e.g. ~/palm/model.git/build/JOBS).
    outbase = str(base / case)

    def resolve(key: str, default_template: str) -> Path:
        """Resolve a path key; default_template uses {base}/{outbase}/{case} tokens."""
        template = raw_paths.get(key, default_template)
        return _resolve_path_template(template, case, domain_suffix,
                                      str(base), outbase).expanduser()

    # join_input defaults to {base}/{case}/OUTPUT
    # All outputs default to {base}/{case}/OUTPUT_*/
    # Optional explicit input override — bypasses join entirely
    input_override: Optional[Path] = None
    if "input" in raw_paths:
        t = raw_paths["input"]
        input_override = _resolve_path_template(t, case, domain_suffix,
                                                str(base), outbase).expanduser()

    return PathsConfig(
        base             = base,
        join_input       = resolve("join_input",  "{outbase}/OUTPUT"),
        output_join      = resolve("output_join",      "{outbase}/OUTPUT_join"),
        output_splitvar  = resolve("output_splitvar",  "{outbase}/OUTPUT_splitvar"),
        output_splitz    = resolve("output_splitz",    "{outbase}/OUTPUT_splitz"),
        output_splittime = resolve("output_splittime", "{outbase}/OUTPUT_splittime"),
        output_coord     = resolve("output_coord",     "{outbase}/OUTPUT_coord"),
        input_override   = input_override,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(cfg: dict, source: Path) -> None:
    errors = []

    if not cfg.get("case"):
        errors.append("'case' is required and must be a non-empty string.")

    if "paths" not in cfg or "base" not in cfg.get("paths", {}):
        errors.append("'paths.base' is required.")

    complevel = cfg.get("complevel", 4)
    if not (1 <= int(complevel) <= 9):
        errors.append(f"'complevel' must be between 1 and 9, got {complevel}.")

    workers = cfg.get("workers", 1)
    if not (1 <= int(workers) <= 32):
        errors.append(f"'workers' must be between 1 and 32, got {workers}.")

    valid_verbosities = {"debug", "info", "warning", "warn"}
    if cfg.get("verbosity", "info").lower() not in valid_verbosities:
        errors.append(f"'verbosity' must be one of {valid_verbosities}.")

    valid_crs = {"utm", "wgs84", "both"}
    crs = cfg.get("steps", {}).get("coord", {}).get("crs", "utm")
    if crs not in valid_crs:
        errors.append(f"'steps.coord.crs' must be one of {valid_crs}, got '{crs}'.")

    splitz = cfg.get("steps", {}).get("splitz", {})
    z_max = splitz.get("z_max", 300.0)
    if float(z_max) < 0:
        errors.append(f"'steps.splitz.z_max' must be >= 0, got {z_max}.")

    valid_conventions = {"dirpart", "filepart", "tmpdir", "filenum",
                         "filepattern", "singlefile"}
    convention = cfg.get("steps", {}).get("join", {}).get("convention", "filenum")
    if convention not in valid_conventions:
        errors.append(f"'steps.join.convention' must be one of {valid_conventions}.")

    if errors:
        msg = (f"Configuration errors in '{source}':\n"
               + "\n".join(f"  - {e}" for e in errors))
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _check_unknown_keys(raw: dict, defaults: dict, log,
                        prefix: str = "") -> list:
    """
    Warn about config keys that no default declares.

    Both configs are deep-merged over the defaults, so an unrecognised key
    is simply carried along and never read — a typo ('celcius', a mis-
    indented step) silently does nothing. Listing them makes that visible.

    Sections whose keys are free-form (paths, which accepts user templates)
    are not descended into, and required keys that carry no default
    ('case', 'paths') are accepted at the top level.
    """
    _FREEFORM = {"paths"}
    _NO_DEFAULT = {"case", "paths"} if not prefix else set()
    unknown = []
    for key, value in raw.items():
        full = f"{prefix}{key}"
        if key in _NO_DEFAULT:
            continue
        if key not in defaults:
            unknown.append(full)
            continue
        if (key not in _FREEFORM and isinstance(value, dict)
                and isinstance(defaults[key], dict)):
            unknown += _check_unknown_keys(value, defaults[key], log,
                                           prefix=f"{full}.")
    if not prefix and unknown:
        log.warning("[config] unknown key(s) ignored: %s - check spelling "
                    "and indentation.", ", ".join(sorted(unknown)))
    return unknown


def load(path: Union[str, Path]) -> Config:
    """Load and validate a YAML config file. Returns Config."""
    source = Path(path).expanduser().resolve()
    log.debug("[config] loading %s", source)

    if not source.is_file():
        raise FileNotFoundError(f"Configuration file not found: '{source}'")

    with source.open() as fh:
        raw = yaml.safe_load(fh) or {}

    log.debug("[config] top-level keys: %s", ", ".join(raw))

    layout = "legacy"
    if is_new_layout(raw):
        layout = "new"
        try:
            raw, notes = translate(raw)
        except LayoutError as exc:
            raise ValueError(f"Configuration error in '{source}':\n"
                             f"  - {exc}") from None
        from .layout import to_new_names
        from .log import set_message_names
        set_message_names(to_new_names)
        for note in notes:
            log.info("[config] %s", note)
    else:
        _check_unknown_keys(raw, _DEFAULTS, log)

    cfg = _deep_merge(_DEFAULTS, raw)
    _validate(cfg, source)

    case   = cfg["case"]
    domain = cfg.get("domain")
    domain_suffix = _parse_domain_suffix(domain)

    log.debug("[config] case %s, domain suffix %s", case,
              domain_suffix or "(root)")

    paths = _build_paths(cfg.get("paths", {}), case, domain_suffix)
    s     = cfg["steps"]

    join = JoinConfig(
        enabled         = bool(s["join"]["enabled"]),
        filelist        = s["join"]["filelist"],
        convention      = str(s["join"]["convention"]),
        file_prefix     = str(s["join"]["file_prefix"]),
        file_suffix     = str(s["join"]["file_suffix"]),
        create_new_file = bool(s["join"]["create_new_file"]),
        offset_min      = int(s["join"]["offset_min"]),
        part_timeshift  = int(s["join"]["part_timeshift"]),
        output_timestep = int(s["join"]["output_timestep"]),
        merge_tol       = (None if s["join"]["merge_tol"] is None
                           else float(s["join"]["merge_tol"])),
        complevel       = int(s["join"]["complevel"]),
    )
    splitvar = SplitvarsConfig(
        enabled      = bool(s["splitvar"]["enabled"]),
        vars_2d      = s["splitvar"]["vars_2d"],
        vars_3d      = s["splitvar"]["vars_3d"],
        group_vars   = list(s["splitvar"]["group_vars"]),
        group_suffix = str(s["splitvar"]["group_suffix"]),
    )
    splitz = SplitzConfig(
        enabled = bool(s["splitz"]["enabled"]),
        z_max   = float(s["splitz"]["z_max"]),
        z_coord = str(s["splitz"]["z_coord"]),
    )
    splittime = SplittimeConfig(
        enabled   = bool(s["splittime"]["enabled"]),
        time_from = s["splittime"]["time_from"],
        time_to   = s["splittime"]["time_to"],
        t_start   = s["splittime"]["t_start"],
        t_end     = s["splittime"]["t_end"],
        timestep  = s["splittime"]["timestep"],
    )
    coord = CoordConfig(
        enabled  = bool(s["coord"]["enabled"]),
        crs      = str(s["coord"]["crs"]),
        utm_zone = s["coord"].get("utm_zone"),
        celsius  = bool(s["coord"].get("celsius", False)),
        temperature_units = dict(s["coord"].get("temperature_units") or {}),
    )
    # The celsius default flipped from true to false in 0.4.0. A config
    # written before that says nothing about it and would silently start
    # producing kelvin, so say so once, loudly, until it is set explicitly.
    if (layout == "legacy" and coord.enabled
            and "celsius" not in raw.get("steps", {}).get("coord", {})):
        log.warning("[config] steps.coord.celsius not set - temperatures "
                    "are in degrees C (default since 0.6.0); set it to "
                    "silence this.")

    config = Config(
        layout    = layout,
        case      = case,
        domain    = domain,
        complevel = int(cfg["complevel"]),
        overwrite = bool(cfg["overwrite"]),
        verbosity = cfg["verbosity"].lower(),
        chain     = bool(cfg.get("chain", True)),
        workers   = int(cfg.get("workers", 1)),
        paths     = paths,
        steps     = StepsConfig(join, splitvar, splitz, splittime, coord),
        _source   = source,
    )

    for name in ("join_input", "output_join", "output_splitvar",
                 "output_splitz", "output_splittime", "output_coord"):
        log.debug("[config] %s: %s", name, getattr(paths, name))

    if join.enabled and not paths.join_input.exists():
        log.warning("[config] join input directory does not exist yet: %s",
                    paths.join_input)

    return config
