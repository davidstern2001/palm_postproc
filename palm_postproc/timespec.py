"""
palm_postproc.timespec
----------------------
Parses the time window of the `time:` config section into seconds since a
file's origin_time.

The same vocabulary palm2gis uses, so one window can be stated once and
mean the same period in a 10-minute 3D file and an hourly surface file:

    '2023-08-24 13:00'   absolute time
    '13:00'              clock time, first occurrence after origin_time
    '6h' / '90min'       offset from origin_time
    3600                 seconds since origin_time

Public API
----------
  to_seconds(spec, origin)  ->  float | None
  parse_origin(attr)        ->  datetime | None
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

_UNITS = {"s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
          "m": 60.0, "min": 60.0, "mins": 60.0, "minute": 60.0,
          "minutes": 60.0,
          "h": 3600.0, "hr": 3600.0, "hrs": 3600.0, "hour": 3600.0,
          "hours": 3600.0,
          "d": 86400.0, "day": 86400.0, "days": 86400.0}

_ABSOLUTE = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S",
             "%Y-%m-%dT%H:%M", "%Y-%m-%d")
_CLOCK = ("%H:%M:%S", "%H:%M")


class TimeSpecError(ValueError):
    pass


def parse_origin(attr):
    """The origin_time attribute as a datetime, or None if unreadable.

    PALM writes a UTC-offset suffix ('2023-08-23 17:00:00 +02'), which is
    dropped: PALM's own seconds are counted from the stated wall clock.
    """
    if not attr:
        return None
    text = re.sub(r"\s*[+-]\d{2}(:?\d{2})?\s*$", "", str(attr).strip())
    for fmt in _ABSOLUTE:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def to_seconds(spec, origin):
    """Seconds since origin_time for one end of the window.

    *origin* is a datetime (from parse_origin) or None; it is needed only
    for an absolute or clock time. None spec -> None (open end).
    """
    if spec is None:
        return None
    if isinstance(spec, bool):
        raise TimeSpecError(f"{spec!r} is not a time.")
    if isinstance(spec, (int, float)):
        return float(spec)
    if isinstance(spec, datetime):
        return _since(spec, origin, spec)

    text = str(spec).strip()
    if not text:
        return None

    # seconds as a bare string
    try:
        return float(text)
    except ValueError:
        pass

    # offset from origin_time, e.g. '6h', '90 min'
    m = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*([a-z]+)", text, re.IGNORECASE)
    if m and m.group(2).lower() in _UNITS:
        return float(m.group(1)) * _UNITS[m.group(2).lower()]

    for fmt in _ABSOLUTE:
        try:
            return _since(datetime.strptime(text, fmt), origin, text)
        except ValueError:
            continue

    # clock time: the first occurrence at or after origin_time
    for fmt in _CLOCK:
        try:
            clock = datetime.strptime(text, fmt).time()
        except ValueError:
            continue
        if origin is None:
            raise TimeSpecError(
                f"'{text}' is a clock time, but the file carries no "
                f"origin_time to resolve it against. Give seconds, an "
                f"offset ('6h') or a full date.")
        when = datetime.combine(origin.date(), clock)
        if when < origin:
            when += timedelta(days=1)
        return (when - origin).total_seconds()

    raise TimeSpecError(
        f"'{text}' is not a time. Use seconds (3600), an offset ('6h', "
        f"'90min'), a clock time ('13:00') or a full date "
        f"('2023-08-24 13:00').")


def _since(when, origin, shown):
    if origin is None:
        raise TimeSpecError(
            f"'{shown}' is an absolute time, but the file carries no "
            f"origin_time to count from. Give seconds or an offset "
            f"('6h') instead.")
    return (when - origin).total_seconds()
