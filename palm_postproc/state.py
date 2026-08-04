"""
palm_postproc.state
-------------------
Lightweight resume system: records a fingerprint for each output file and
skips re-processing when the source file and config are unchanged.

State is stored as JSON in {base}/{case}/.palm_postproc_state.json.

Fingerprint = (source_path, source_mtime, source_size, config_hash)

Usage
-----
    state = State.load(cfg)
    if state.is_current(src_path, out_path, cfg_hash):
        log.info("Up to date, skipping: %s", out_path.name)
        return True
    # ... process ...
    state.mark_done(src_path, out_path, cfg_hash)
    state.save()
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("palm_postproc")

_STATE_FILENAME = ".palm_postproc_state.json"


# ---------------------------------------------------------------------------
# Config hashing
# ---------------------------------------------------------------------------

def config_hash(cfg) -> str:
    """
    Return a short hex digest that changes whenever any config value that
    affects output changes.  Deliberately excludes paths.base and verbosity.
    """
    import dataclasses
    # Collect all step parameters into a stable dict
    relevant = {
        "case":      cfg.case,
        "domain":    cfg.domain,
        "complevel": cfg.complevel,
        "splitvar": dataclasses.asdict(cfg.steps.splitvar),
        "splitz":   dataclasses.asdict(cfg.steps.splitz),
        "splittime": dataclasses.asdict(cfg.steps.splittime),
        "coord":    dataclasses.asdict(cfg.steps.coord),
    }
    blob = json.dumps(relevant, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def join_config_hash(cfg) -> str:
    """
    Digest of the settings that change what the JOIN step produces.

    join is deliberately excluded from config_hash() above: its outputs are
    the *input* to the post-join chain, so mixing them would invalidate
    every downstream file whenever an unrelated join setting changed.
    But join still has to notice its own settings changing — offset_min,
    part_timeshift and output_timestep all alter the data — otherwise the
    plain file-exists check silently keeps a stale joined file.
    """
    import dataclasses
    j = dataclasses.asdict(cfg.steps.join)
    j.pop("enabled", None)          # toggling the step is not a data change
    relevant = {"case": cfg.case, "domain": cfg.domain, "join": j}
    blob = json.dumps(relevant, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Per-file record
# ---------------------------------------------------------------------------

@dataclass
class _Record:
    src_path:    str
    src_mtime:   float
    src_size:    int
    cfg_hash:    str
    finished_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "src_path":    self.src_path,
            "src_mtime":   self.src_mtime,
            "src_size":    self.src_size,
            "cfg_hash":    self.cfg_hash,
            "finished_at": self.finished_at,
        }

    @staticmethod
    def from_dict(d: dict) -> "_Record":
        return _Record(
            src_path    = d["src_path"],
            src_mtime   = d["src_mtime"],
            src_size    = d["src_size"],
            cfg_hash    = d["cfg_hash"],
            finished_at = d.get("finished_at", 0.0),
        )


def _file_fingerprint(path: Path) -> tuple[float, int]:
    """Return (mtime, size) for a file."""
    st = path.stat()
    return st.st_mtime, st.st_size


# ---------------------------------------------------------------------------
# State object
# ---------------------------------------------------------------------------

class State:
    """
    Manages the resume state for a single pipeline run.

    The state file key is the *output* file path (absolute str).
    This means the same output file cannot be produced by two different
    source/config combinations simultaneously — which is what we want.
    """

    def __init__(self, state_path: Path, records: dict[str, _Record],
                 meta: dict | None = None):
        self._path    = state_path
        self._records = records
        self._meta    = dict(meta or {})
        self._dirty   = False

    # ---- Construction -----------------------------------------------------

    @classmethod
    def load(cls, cfg) -> "State":
        state_path = Path(cfg.paths.base) / cfg.case / _STATE_FILENAME
        records: dict[str, _Record] = {}
        meta: dict = {}

        if state_path.is_file():
            try:
                with state_path.open() as fh:
                    raw = json.load(fh)
                meta = raw.pop(cls._META_KEY, {}) or {}
                records = {k: _Record.from_dict(v) for k, v in raw.items()}
                log.debug("[state] Loaded %d record(s) from %s",
                          len(records), state_path.name)
            except Exception as exc:
                log.warning("[state] Could not read state file (%s) — starting fresh.", exc)

        return cls(state_path, records, meta)

    # ---- Query ------------------------------------------------------------

    def is_current(
        self,
        src_path: Path,
        out_path: Path,
        cfg_hash: str,
    ) -> bool:
        """
        Return True if *out_path* is up to date:
          - output file exists on disk
          - was produced from *src_path* at its current mtime/size
          - was produced with the same config hash
        """
        if not out_path.exists():
            return False

        key = str(out_path.resolve())
        rec = self._records.get(key)
        if rec is None:
            return False

        if rec.cfg_hash != cfg_hash:
            log.debug("[state] Config changed for %s — will reprocess.", out_path.name)
            return False

        if not src_path.exists():
            return False

        mtime, size = _file_fingerprint(src_path)
        if rec.src_mtime != mtime or rec.src_size != size:
            log.debug("[state] Source changed for %s — will reprocess.", out_path.name)
            return False

        return True

    # ---- Update -----------------------------------------------------------

    def mark_done(
        self,
        src_path: Path,
        out_path: Path,
        cfg_hash: str,
    ) -> None:
        """Record that *out_path* was successfully produced from *src_path*."""
        mtime, size = _file_fingerprint(src_path)
        key = str(out_path.resolve())
        self._records[key] = _Record(
            src_path  = str(src_path.resolve()),
            src_mtime = mtime,
            src_size  = size,
            cfg_hash  = cfg_hash,
        )
        self._dirty = True

    def invalidate(self, out_path: Path) -> None:
        """Remove any record for *out_path* (e.g. after a failed write)."""
        key = str(out_path.resolve())
        if key in self._records:
            del self._records[key]
            self._dirty = True

    # ---- Parallel-safe transfer -------------------------------------------

    def records(self) -> dict:
        """Return this State's records (for transfer from a worker process)."""
        return dict(self._records)

    def merge(self, records: dict) -> None:
        """
        Merge records produced elsewhere (e.g. by a worker process) into
        this State.

        Each worker holds its own State object, so a worker calling save()
        would write only the files IT processed and drop every record the
        other workers wrote concurrently — last writer wins, and the resume
        information is silently lost. Workers therefore return their records
        and only the parent process merges and saves.
        """
        if not records:
            return
        self._records.update(records)
        self._dirty = True

    # ---- Run-level metadata -----------------------------------------------
    # Stored under a reserved key so it travels with the per-file records.

    _META_KEY = "__meta__"

    def get_meta(self, name: str, default=None):
        meta = self._meta
        return meta.get(name, default)

    def set_meta(self, name: str, value) -> None:
        if self._meta.get(name) == value:
            return
        self._meta[name] = value
        self._dirty = True

    # ---- Persistence ------------------------------------------------------

    def save(self) -> None:
        """Write state to disk (atomic via temp file + rename)."""
        if not self._dirty:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            with tmp.open("w") as fh:
                payload = {k: v.to_dict() for k, v in self._records.items()}
                if self._meta:
                    payload[self._META_KEY] = self._meta
                json.dump(payload, fh, indent=2)
            tmp.replace(self._path)
            log.debug("[state] Saved %d record(s) to %s",
                      len(self._records), self._path.name)
            self._dirty = False
        except Exception as exc:
            log.warning("[state] Could not save state file: %s", exc)
