# palm_postproc

Post-processing pipeline for PALM large-eddy simulation NetCDF output.

**Authors:** This code was written by [Claude.ai](https://claude.ai), configured and directed by David Stern Mgr. (IPR Praha).

---

## Overview

`palm_postproc` orchestrates four sequential processing steps on PALM `_av_xy` and `_av_3d` output files:

| Step | Script | What it does |
|------|--------|--------------|
| `splitvar` | `steps/splitvar.py` | Splits into one NetCDF per variable |
| `splitz` | `steps/splitz.py` | Restricts 3D files to z ≤ z_max |
| `splittime` | `steps/splittime.py` | Subsamples or slices the time dimension |
| `coord` | `steps/coord.py` | Appends UTM / WGS84 georeferenced coordinates |

Each step can be individually enabled or disabled in the config file.

---

## Directory layout

```
palm_postproc/
├── palm_postproc/          # installable package
│   ├── __init__.py
│   ├── config.py
│   ├── log.py
│   ├── pipeline.py
│   └── steps/
│       ├── splitvar.py
│       ├── splitz.py
│       ├── splittime.py
│       └── coord.py
├── config/                # one YAML per job — user-managed
│   └── holesovice_now.yaml
├── run_postproc.py         # entry point
├── test.py                 # synthetic self-test (no real data needed)
├── template.yaml           # fully-commented config template
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 1b. Check that the environment works. This builds a small synthetic PALM
#     output tree in a temporary directory, runs the pipeline over it in
#     chain, classic, --only and dry-run modes, and verifies the outputs —
#     no real data needed, a few seconds.
python test.py

# 2. Copy PALM output files into the job's OUTPUT directory
cp /path/to/palm/job/OUTPUT/*.nc holesovice_stern_now/OUTPUT/

# 3. Create a config file from the template
cp template.yaml config/holesovice_now.yaml
# edit config/holesovice_now.yaml as needed

# 4. Run
python run_postproc.py -c config/holesovice_now.yaml

# Dry-run (validates config and prints plan, no files written)
python run_postproc.py -c config/holesovice_now.yaml --dry-run

# Debug output
python run_postproc.py -c config/holesovice_now.yaml -v
```

---

## Config reference

See `template.yaml` for the full annotated configuration reference.

Key options:

```yaml
case: holesovice_stern_now   # required
paths:
  base: holesovice_stern_now

steps:
  splitvar:
    enabled: true
    vars_2d: all             # or a list of variable names
    vars_3d: all
  splitz:
    enabled: true
    z_max: 300.0
  splittime:
    enabled: true
    timestep: "1h"
  coord:
    enabled: true
    crs: utm                 # utm | wgs84 | both
    celsius: true            # K -> degrees C for theta*/tsurf* (default true)
```

---

## CLI flags

| Flag | Description |
|------|-------------|
| `-c CONFIG` | Path to YAML config file (required) |
| `-n / --dry-run` | Validate and print plan, write nothing |
| `-v / --verbose` | DEBUG-level logging |
| `-q / --quiet` | WARNING-level logging only |
| `--log-datetime` | Prepend timestamps to log lines |
| `--log-file PATH` | Also append the log to a file (uncoloured, always timestamped) |
| `--workers N` | Override `workers` from the config (1–32) |
| `--version` | Print the version and exit |

---

## Log output

The log uses the same PALM-GeM style as `palm_preproc` and `palm2gis`, so
the three read as one toolchain:

```
14:03:17  Joining PALM output parts               <- step (bold)
14:03:17    [join] 4 file(s) to join              <- detail (indented)
14:03:17    [join]   phase 2/3: analysing         <- debug (grey, -v only)
14:03:17  WARNING  [join] discontinuous timestep  <- warnings/errors tagged
```

Colour is dropped automatically when stderr is not a terminal, so a
captured batch log stays readable. `--log-datetime` switches the timestamp
to a full date; `--log-file` always writes one.

---

## Shared conventions with the companion tool

`palm_postproc` and `palm2gis` share a `join` step and then diverge:

```
PALM  ->  join  -+->  splitvar -> splitz -> splittime -> coord   (analysis)
                 +->  palm2gis  (+ static driver + _p3d)         (GIS)
```

**`join` output IS the intended feed for palm2gis.** `OUTPUT_join` keeps raw
PALM geometry and units while stitching the restart-cycle parts together,
which is exactly what palm2gis wants — it applies `origin_x` / `origin_y` /
`origin_z` and the unit handling itself. Joined files are stamped
`palm_postproc_stage = "join"`, which palm2gis reads as confirmation. See
`config/join_only.yaml`.

**`coord` output must NEVER be given to palm2gis.** Coordinates and units
would be applied twice; palm2gis rejects such files at startup.

Both branches come from one run: `OUTPUT_join` survives even in chain mode,
so the analysis chain and the GIS feed do not require separate invocations.

| Directory | palm2gis can read it | Note |
|---|---|---|
| `OUTPUT/`, `OUTPUT_join/` | yes, all stages | the intended feed |
| `OUTPUT_splitvar/` | voxel, wind | one variable per file; surface files never appear here |
| `OUTPUT_splitz/`, `OUTPUT_splittime/` | voxel, wind | already cut; palm2gis warns and resolves `bounds.zmax` against what survived |
| `OUTPUT_coord/` | **never** | origin and units applied twice |

Because the two carry the same conventions independently, these settings
must be kept in step **by hand**. There is no shared config.

| Convention | `palm_postproc` | `palm2gis` | Note |
|---|---|---|---|
| CRS | `steps.coord.utm_zone`, default EPSG:32633 | `crs.palm`, default `EPSG:32633` | Change **both** for a non-UTM-33N domain (e.g. S-JTSK EPSG:5514), or the branches disagree |
| Celsius | `steps.coord.celsius`, default **false** | `celsius`, default **false** | Aligned in 0.4.0 (it used to be true here). Both default to kelvin |
| Temperature variables identified | `theta*`, `tsurf*`, `t_surf*`, `ta*` | same | Defined in `steps/coord.py::TEMP_PREFIXES` and `palm2gis/steps/thermo.py::TEMP_PREFIXES` — **edit both together** |
| Temperature **scale** | read from each variable's `units` | same | **PALM writes `ta` and `ta_2m*` in °C; `theta`, `tsurf*`, `t_surf*` are kelvin.** Assuming kelvin by name is what produced voxels at −253 °C. PALM's truncated `"degree_"` counts as Celsius. An unrecognised unit **raises** rather than being guessed — override with `steps.coord.temperature_units` / `temperature_units` |
| Potential temperature | never converted | never converted | `theta` is exempt from `celsius`: it is a kelvin-defined quantity |
| Vertical datum | `coord` adds `origin_z` to z | `Grid.abs_z` adds `origin_z` to every output z | Both emit height above sea level. A static driver with `origin_z = 0` warns |
| Input expected | raw / joined PALM output | raw / joined PALM output **+ static driver** | Never each other's output |
| Time origin | `origin_time` attribute, CF epoch honoured incl. UTC offset | `origin_time` attribute, or `domain.origin_time` override | |

The two tools are separate packages by design and share no code, so these
conventions are kept in step **by hand**. If you change the CRS, the Celsius
setting, or the temperature-variable list, change it in both repositories in
the same sitting — this table is the only thing enforcing it.
