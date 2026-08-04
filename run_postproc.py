#!/usr/bin/env python3
"""
run_postproc.py — palm_postproc entry point
============================================
Usage
-----
  python run_postproc.py -c configs/holesovice_now.yaml
  python run_postproc.py -c configs/holesovice_now.yaml --dry-run
  python run_postproc.py -c configs/holesovice_now.yaml -v
  python run_postproc.py -c configs/holesovice_now.yaml -q
  python run_postproc.py -c configs/holesovice_now.yaml --only coord
  python run_postproc.py -c configs/holesovice_now.yaml --only splitvar splitz
  python run_postproc.py -c configs/holesovice_now.yaml --workers 4
  python run_postproc.py -c configs/holesovice_now.yaml --log-file run.log
"""

import argparse
import sys
import time
from pathlib import Path

from palm_postproc.log import setup_logging, step

_VALID_STEPS = ("join", "splitvar", "splitz", "splittime", "coord")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_postproc.py",
        description="PALM output post-processing pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("-c", "--config", required=True, type=Path, metavar="CONFIG",
                   help="Path to the YAML configuration file.")
    p.add_argument("-n", "--dry-run", action="store_true",
                   help="Validate config and print plan without writing files.")
    p.add_argument("--only", nargs="+", metavar="STEP", choices=_VALID_STEPS,
                   help=(f"Run only the specified step(s). "
                         f"Choices: {', '.join(_VALID_STEPS)}"))

    verbosity = p.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true",
                           help="Enable DEBUG-level logging.")
    verbosity.add_argument("-q", "--quiet", action="store_true",
                           help="WARNING-level logging only.")
    p.add_argument("--log-datetime", action="store_true",
                   help="Prepend timestamps to log lines.")
    p.add_argument("--log-file", type=Path, default=None, metavar="PATH",
                   help=("Also append the log to this file (uncoloured, "
                         "always timestamped). Useful for queued jobs."))
    p.add_argument("--workers", type=int, default=None, metavar="N",
                   help=("Override steps' parallel worker count from the "
                         "config (1-32)."))
    p.add_argument("--version", action="store_true",
                   help="Print the palm_postproc version and exit.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    from palm_postproc import __version__

    if args.version:
        print(f"palm_postproc {__version__}")
        return 0

    if args.verbose:
        verbosity = "debug"
    elif args.quiet:
        verbosity = "warning"
    else:
        verbosity = "info"

    log = setup_logging(verbosity=verbosity, log_datetime=args.log_datetime,
                        log_file=args.log_file)

    # The PALM-GeM style has no banner rule: the version line is a plain
    # detail line, exactly as in palm_preproc and palm2gis.
    log.info("palm_postproc %s", __version__)

    # ---- Load config -------------------------------------------------------
    try:
        from palm_postproc.config import load as load_config
        cfg = load_config(args.config)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1
    except ValueError as exc:
        log.error("%s", exc)
        return 1
    except Exception as exc:
        log.error("Unexpected error loading config: %s", exc)
        log.debug("", exc_info=True)
        return 1

    if not args.verbose and not args.quiet:
        log = setup_logging(verbosity=cfg.verbosity,
                            log_datetime=args.log_datetime,
                            log_file=args.log_file)

    # --workers overrides the config, so a run can be throttled on a login
    # node without editing the YAML.
    if args.workers is not None:
        if not (1 <= args.workers <= 32):
            log.error("--workers must be between 1 and 32, got %d", args.workers)
            return 1
        if args.workers != cfg.workers:
            log.info("Workers overridden on the command line: %d -> %d",
                     cfg.workers, args.workers)
        cfg.workers = args.workers

    # ---- Summary -----------------------------------------------------------
    # Show the effective input the pipeline will actually read from
    from palm_postproc.pipeline import _chained_input_dir
    if cfg.paths.input_override is not None:
        effective_input = cfg.paths.input_override
        input_label     = "input (override)"
    elif cfg.steps.join.enabled:
        effective_input = cfg.paths.join_input
        input_label     = "raw input"
    else:
        effective_input = cfg.paths.join_input
        input_label     = "input dir"

    step(log, "Reading configuration")
    log.info("case: %s", cfg.case)
    log.info("domain: %s", cfg.domain or "(root)")
    log.info("%s: %s", input_label, effective_input)
    log.info("complevel: %d", cfg.complevel)
    log.info("chain mode: %s", cfg.chain)
    log.info("workers: %d", cfg.workers)
    log.info("overwrite: %s", cfg.overwrite)
    log.info("dry run: %s", args.dry_run)

    if args.only:
        log.info("--only: %s", ", ".join(args.only))

    log.debug("Steps enabled:")
    for name in _VALID_STEPS:
        log.debug("  %-10s: %s", name, getattr(cfg.steps, name).enabled)

    # ---- Validate input directory -----------------------------------------
    # When join is enabled, validate the raw input; otherwise validate output_join
    check_dir = cfg.paths.join_input
    if not check_dir.is_dir():
        log.error("Input directory does not exist: %s", check_dir)
        return 1

    nc_files = sorted(check_dir.glob("*.nc"))
    # For filenum convention, look for *.000.nc or any .nc
    if not nc_files:
        # Try without suffix for filenum convention
        all_files = list(check_dir.iterdir())
        log.warning("No .nc files found in %s (%d items total)",
                    check_dir, len(all_files))
    else:
        log.info("input files: %d file(s) found in %s",
                 len(nc_files), check_dir.name)
        for f in nc_files:
            log.debug("  %s", f.name)

    # ---- Disabled step warnings -------------------------------------------
    disabled = [n for n in _VALID_STEPS if not getattr(cfg.steps, n).enabled]
    if disabled:
        log.warning("Disabled steps: %s", ", ".join(disabled))

    # ---- --only validation ------------------------------------------------
    only = set(args.only) if args.only else None
    if only:
        order = list(_VALID_STEPS)
        first = min(order.index(s) for s in only)
        if first > 0:
            log.warning("--only %s: make sure upstream output directories exist.",
                        " ".join(sorted(only)))

    # ---- Run ---------------------------------------------------------------
    t0 = time.monotonic()

    try:
        from palm_postproc.pipeline import run as run_pipeline
        run_pipeline(cfg, dry_run=args.dry_run, only=only)
    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
        return 130
    except Exception as exc:
        log.error("Pipeline failed: %s", exc)
        log.debug("", exc_info=True)
        return 1

    step(log, "palm_postproc finished OK ({:.1f}s)",
         time.monotonic() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
