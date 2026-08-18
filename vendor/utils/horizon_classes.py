"""Bucket ``time_horizon_months`` values into coarse temporal-scale classes."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Conversion factors copied from `temporal_manifolds.horizon.cache.UNIT_TO_MONTHS`
# in the temporal-manifolds-last-position repo, which defines a month as the
# mean Gregorian month of 30.4375 days.
UNIT_TO_MONTHS = {
    "second": 1 / (30.4375 * 86400),
    "minute": 1 / (30.4375 * 1440),
    "hour": 1 / (30.4375 * 24),
    "day": 1 / 30.4375,
    "week": 7 / 30.4375,
    "month": 1.0,
    "year": 12.0,
    "decade": 120.0,
    "century": 1200.0,
}

# Half-open [lower, upper) buckets, in ascending order.
HORIZON_CLASS_UNITS = (
    ("second", "minute"),
    ("minute", "hour"),
    ("hour", "day"),
    ("day", "week"),
    ("week", "month"),
    ("month", "year"),
    ("year", "decade"),
    ("decade", "century"),
    ("century", None),
)
HORIZON_CLASS_LABELS = tuple(
    f"{lower} - {upper}" if upper else f"{lower} - +inf"
    for lower, upper in HORIZON_CLASS_UNITS
)
# Edges in months: [1 second, 1 minute, ..., 1 decade, +inf].
HORIZON_CLASS_EDGES = np.array(
    [UNIT_TO_MONTHS[lower] for lower, _ in HORIZON_CLASS_UNITS] + [np.inf],
    dtype=np.float64,
)
HORIZON_CLASS_COUNT = len(HORIZON_CLASS_UNITS)


# The stored horizons are rounded to six significant figures, so a value meant to
# sit exactly on an edge can land just below it. Compare with this relative slack.
EDGE_TOLERANCE = 1e-5


def horizon_class(
    months: np.ndarray | pd.Series | list[float], *, tolerance: float = EDGE_TOLERANCE
) -> np.ndarray:
    """Return the integer class of each horizon, or -1 when it falls outside.

    Class ``i`` covers ``[edges[i], edges[i + 1])``. Horizons shorter than one
    second, and non-finite values, are labelled ``-1``.
    """

    values = pd.to_numeric(pd.Series(months), errors="coerce").to_numpy(dtype=np.float64)
    # Nudge upward so a rounded-down value still reaches its edge; a value already
    # on the edge stays in the upper bucket, which `side="right"` also gives it.
    nudged = values * (1.0 + tolerance)
    classes = np.searchsorted(HORIZON_CLASS_EDGES, nudged, side="right") - 1
    classes = np.where(
        np.isfinite(values) & (nudged >= HORIZON_CLASS_EDGES[0]), classes, -1
    )
    return classes.astype(np.int64)


def horizon_class_label(
    months: np.ndarray | pd.Series | list[float], *, tolerance: float = EDGE_TOLERANCE
) -> np.ndarray:
    """Human-readable label for each horizon; ``"< 1 second"`` when out of range."""

    classes = horizon_class(months, tolerance=tolerance)
    labels = np.array(HORIZON_CLASS_LABELS + ("< 1 second",), dtype=object)
    return labels[np.where(classes < 0, HORIZON_CLASS_COUNT, classes)]
