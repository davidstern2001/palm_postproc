# Changelog

## 0.6.0

**Config layout, shared with palm2gis and palm_preproc**

- A config now says what it wants out, not which internal steps run:
  `project`, `input`, `variables`, `region`, `time`, `joined`, `analysis`
  and `advanced`. `case`, `paths`, `steps`, `chain`, `complevel`,
  `workers` and `verbosity` are gone from the new layout; the five steps
  are switched on by the blocks that need them (no `region.z_max` means
  no vertical cut, `analysis.dir: null` leaves only the joined files).
- `time.from` / `time.to` accept palm2gis's vocabulary - an absolute time,
  a clock time, an offset ('6h') or seconds since origin_time - and are
  resolved against each file's own axis, so one window means the same
  period in a 10-minute 3D file and an hourly surface file. The pre-0.6
  record indices (`t_start` / `t_end`) still work.
- `steps.coord.celsius` is replaced by `advanced.units.temperature`
  (`K` or `C`), and **the default changes to `C`**, matching palm2gis
  0.25.0. `advanced.units.potential_temperature` accepts only `K`.
  `temperature_units` becomes `advanced.units.overrides`.
- `steps.coord.utm_zone` (an EPSG code) becomes `input.crs`
  ('EPSG:32633'), `chain: false` becomes
  `advanced.performance.keep_intermediate: true`, and the join settings
  move under `advanced.join`.
- An unknown or misspelt key in the new layout stops the run with a
  suggestion instead of being warned about and ignored. Log messages use
  the new names.
- Configs in the pre-0.6 layout still load unchanged, except that they
  too get the new Celsius default.

## 0.5.1

**Logging - same style as palm2gis and palm_preproc**

- One `[step] wrote NAME: ...` line per output instead of a
  `Processing a -> b ...` line followed by a `✓ size (time)` line; the
  start-of-file lines are debug only.
- Existing outputs are reported as `[step] kept (exists): NAME` at info
  (was a warning), dry runs as `[step] would write NAME (dry run)`.
- ASCII only (`-`, `->`), lowercase messages after the `[tag]`, no
  `[step] Done.` lines.
- The configuration summary is three lines (case/domain, input, steps);
  the remaining settings are debug. `overwrite: true` now warns, as in
  palm2gis.
- Long warnings shortened (celsius default, unknown keys, UTM 33N
  fallback, the join part-mismatch error); the explanations moved to
  code comments.
- join's per-dimension / per-variable debug lines collapsed into one.

## 0.5.0

**join can no longer silently corrupt a file**

- Every time-invariant variable is compared across parts before anything is
  written. Phase 1 takes them from `parts[0]` alone and Phase 3 then appends
  every part's data along time, which is correct for restart-cycle parts and
  silently wrong for parts holding a different piece of space: one part's
  `xs`/`ys`/`zs` paired with another part's values. Nothing downstream could
  detect it — the file was structurally valid and passed every check. Such a
  set of parts is now rejected, naming the variable that differs.

**Temperature scale is read from `units`, not guessed from the name**

- PALM writes `ta` and `ta_2m*` in degrees Celsius while `theta`, `tsurf*`
  and `t_surf*` are kelvin, so `celsius: true` drove them 273.15 below
  reality. `celsius` now selects the OUTPUT scale, the input scale comes from
  each variable's `units` (PALM's truncated `"degree_"` included), and
  conversion runs only when the two differ. `theta` is exempt: potential
  temperature stays kelvin. An unrecognised unit **raises** rather than being
  guessed — `steps.coord.temperature_units` is the override.

**Provenance and the palm2gis handoff**

- Joined files carry provenance for the first time — previously an
  `OUTPUT_join` directory was the one anonymous output of this tool. They are
  stamped `palm_postproc_stage = "join"`, which palm2gis reads as positive
  confirmation that a file is safe to consume rather than inferring it from
  the absence of `coord`'s fingerprints.
- `splitz` records `palm_postproc_z_max` / `z_coord` / `z_levels` and
  `splittime` records its window. A cut file is otherwise indistinguishable
  from a full one, so a consumer deriving its own range from `nz`/`dz`
  re-cut an already-cut axis with no warning anywhere.
- `config/join_only.yaml` documents the join-only run that feeds palm2gis.
- Near-duplicate timestep merges are reported at info with the tolerance
  used; they change the time axis a consumer will read.

**Other**

- `configs/` renamed to `config/`, matching palm2gis.
- README: the two tools are no longer described as branches that never
  consume each other's output. `join` output IS the intended palm2gis feed;
  `coord` output never is. Adds the directory table and the temperature-scale
  and vertical-datum conventions.

## 0.4.1

**Correctness**

- `join` no longer fails with
  `IndexError: size of data array does not conform to slice` when the parts
  of one output file do not tile the global time axis end to end.

  Each part was matched to the global timestep list as a single contiguous
  run (`ptsmin`/`ptsmax` → `tsmin`/`tsmax`) and written with one slice
  assignment per variable. That is only correct when no other part
  contributes a timestep that falls *between* two of this part's
  timesteps. A re-run restart cycle whose output times are offset from the
  cycle it overlaps — or two cycles written at different
  `dt_do3d`/`averaging_interval` — breaks the assumption: the destination
  slice comes out longer than the source data and netCDF4 rejects the
  write. The failure surfaces on the first affected file and aborts the
  whole pipeline.

  Parts are now matched timestep by timestep. Each part carries an explicit
  `(source index → global index)` map and is written one record at a time,
  so parts may overlap, interleave, or leave gaps in any combination. Later
  parts still win on collision, as before.

- Near-duplicate timesteps are now merged whatever `output_timestep` is set
  to. The merge test was `abs(dt) < output_timestep / 2.0`, which with the
  default `output_timestep: 0` compares against zero and therefore never
  fires, so the sub-second differences PALM writes at restart-cycle
  boundaries survived into the joined file as separate records. The
  tolerance now falls back to a quarter of the median output interval,
  capped at 1 s, and can be set explicitly with the new
  `steps.join.merge_tol` key. The cap keeps an irregular time axis safe:
  a quarter of the median alone comes out at 887 s on
  `t = [0, 100, 3600, 7200]` and would merge two genuine records.

- `time` is no longer written twice per part. It was written once with
  `part_timeshift` applied and then again, unshifted, by the generic
  time-dependent-variable loop, so a non-zero `part_timeshift` was silently
  discarded. The generic loop now skips it.

- A part whose variable is *smaller* than the joined output in a non-time
  dimension is reported and skipped instead of raising a bare
  `ValueError: operands could not be broadcast together`. This is what a
  mid-run change of `nz_do3d` looks like on disk.

- A file whose timesteps are all removed by the `output_timestep` filter
  closes cleanly with a warning instead of raising `IndexError` on an empty
  timestep list.

- When two parts hold the same instant, the later part now wins for the
  *value* of `time` as well as for the data, so the two can no longer come
  from different parts.

- A timestamp repeated *within* one part now resolves to the last of the
  repeats rather than the first, matching the cross-part rule. This is the
  one behaviour change here that is not a bug fix; PALM should not emit
  such a file, but if yours does, the record kept is now the later one.

**Config**

- New `steps.join.merge_tol` (seconds, default unset = auto). Two timesteps
  closer together than this count as the same instant.

**Tests**

- `test.py` gains a join regression case built from three parts, the third
  of which interleaves with the second, plus unit checks for
  `_merge_tolerance`.

## 0.4.0

**Logging**

- The log now uses the PALM-GeM style shared with `palm_preproc` and
  `palm2gis`: timestamped lines, bold step announcements, indented detail
  lines, and level tags only on warnings and errors. It replaces the
  `[INFO ]`/`[DEBUG]` tag on every line, the `====` banner rule and the
  `--- Step: x ---` separators, so the three tools read as one toolchain in
  a terminal and in a batch log. The formatter is character-for-character
  the same as the other two.

  The step modules keep their printf-style calls (`log.info("%d file(s)",
  n)`); the formatter renders whatever the record produces, so nothing had
  to be rewritten to change the appearance. `progress()`/`info()`/
  `debug()`/`warning()`/`error()` helpers with lazy `{}` formatting are
  available for new code, and `step(log, ...)` is the `progress()`
  equivalent for the step modules, which are handed a logger.

  The config summary keys are lowercase and unpadded to match the detail
  lines elsewhere (`case: selftest`, not `Case        : selftest`).

**Correctness**

- `chain: false` and `--only splitvar` no longer crash. `splitvar.run()`
  read `cfg.paths.input`, which is not a field of `PathsConfig` (the
  fields are `join_input`, `output_join` and `input_override`), so every
  classic-mode run failed immediately with
  `AttributeError: 'PathsConfig' object has no attribute 'input'`. It now
  uses the same `_chained_input_dir()` resolver the chained path uses.
  Only the chained pipeline was ever exercised, which is why this survived.

- `splitz` slices **every** vertical coordinate the file carries, not only
  the configured `z_coord`. `splitvar` groups `u`, `v` and `w` into a single
  `.uvw` file where u/v live on `zu_3d` and w on the staggered `zw_3d`;
  slicing only `zu_3d` left `w` at full column height in that file, so the
  group file mixed a 300 m scalar field with a full-depth `w`.

- The kelvin -> degrees C conversion now covers the same variables as
  `palm2gis`: `theta*`, `tsurf*`, `t_surf*` and `ta*`. It previously covered
  only `theta*` and `tsurf*`, so `ta_2m*` and `t_surf` stayed in kelvin
  beside their degrees-C neighbours and an `OUTPUT_coord` directory could
  mix the two units silently.

- Time-invariant variables (`ind_z_xy`, `zusi`, `zwwi`) no longer emit a
  `UserWarning` on every write. They inherit the parent dataset's encoding,
  which still declares `time` unlimited; the stale `unlimited_dims` entry is
  now dropped before writing.

**Breaking**

- `steps.coord.celsius` now defaults to **false** (kelvin), matching
  `palm2gis`. It used to default to true here and false there, so the two
  branches of the same run disagreed and neither default told you what the
  other had done. False is the safer choice: raw PALM fidelity and standard
  CF units. **A config that does not set `celsius` explicitly will now
  produce kelvin where 0.3.0 produced degrees C** — a warning naming the
  change is logged whenever the key is absent. Set `celsius: true` to keep
  the old behaviour.

**Performance**

- The kelvin -> degrees C conversion is lazy. It used `da.values` and
  `np.isfinite(da.values)`, materialising the full array twice per variable
  and defeating the dask chunking the file was opened with; a chunked 3D
  file is now converted chunk by chunk at write time.

**Features**

- Outputs carry provenance: `palm_postproc_version`,
  `palm_postproc_config`, `palm_postproc_config_sha`, `palm_postproc_case`,
  `palm_postproc_command`, `palm_postproc_processed` and
  `palm_postproc_source`, plus a CF-style `history` line appended to (never
  replacing) any history the file already had. An `OUTPUT_coord` directory
  is otherwise anonymous six months later.

- `--log-file PATH` appends the full log, uncoloured and always timestamped,
  to a file as well as stderr — a queued job otherwise leaves no log unless
  the batch system captures stderr.

- `--workers N` overrides `workers` from the config, so a run can be
  throttled on a login node without editing the YAML.

- `--version` prints the version and exits; the version is also logged in
  the startup banner instead of only at DEBUG level.

- `test.py`: builds a synthetic PALM output tree in a temporary directory,
  runs the pipeline over it in chain mode, classic mode, `--only` mode and
  dry-run mode, and verifies the results. Run it to check a new environment.
  It covers the join timestep union, variable splitting, both vertical
  coordinates in `splitz`, the UTM/WGS84 coordinates, the epoch shift, the
  temperature conversion including fill-value protection, resume, and the
  provenance attributes.

**Maintenance**

- The package uses relative imports (`from ..config import Config`) like
  `palm_preproc` and `palm2gis` do. It previously used absolute ones, which
  only resolved because the repository root happened to be on `sys.path`
  via the current working directory; running `run_postproc.py` from
  anywhere else broke.

- Added `LICENSE` (MIT, as in `palm_preproc`) and `.gitignore`.

## 0.3.0

**Features**

- `steps.coord.crs` is now actually implemented. `wgs84` and `both` add
  CF-compliant 2-D auxiliary `latitude`/`longitude` coordinates on (y, x),
  referenced from each data variable's `coordinates` attribute. Previously
  both values silently produced UTM-only output.

**Correctness**

- `coord` no longer snaps timestamps to the averaging interval when the
  file is not an averaged product. Rounding to `dt_averaging` (which
  defaulted to 600 s even when absent) moved genuine timestamps of
  instantaneous output onto a grid they were never on.
- `coord` no longer assumes the file's CF epoch equals `origin_time`. The
  epoch is parsed and the difference applied, so a restarted run with a
  different epoch gets correct times. Any UTC offset in the units string
  is honoured rather than stripped, matching what `xarray.decode_cf` does,
  so the decoded and numeric paths now agree exactly.
- A missing EPSG is reported at WARNING level instead of DEBUG. The
  EPSG:32633 fallback is a guess and a domain in another CRS was
  previously mislabelled silently.
- Parallel workers no longer clobber the resume state. Each worker held
  its own `State` and saved the whole record set, so the last writer
  discarded everything the others had recorded — resume silently forgot
  completed files exactly when `workers > 1` made it worth having. Workers
  now return their records and only the parent merges and saves.
- `join` reruns when its own settings change. Its outputs are guarded by a
  file-exists check and `join` is excluded from the downstream config
  hash, so changing `offset_min`, `part_timeshift` or `output_timestep`
  previously left stale joined files in place. A separate join config
  hash is stored in the state file and a change forces a rejoin.

**Usability**

- Unknown configuration keys are listed in a warning instead of being
  silently ignored, so a typo or a mis-indented setting is visible.
- README documents the shared conventions with `palm2gis`, including the
  known divergence in which temperature variables each tool converts.

