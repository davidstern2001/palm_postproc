# Changelog

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

