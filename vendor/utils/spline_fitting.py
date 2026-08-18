"""Fit a planar spline through the PLS1-PLS2 point cloud.

The saved `ctype_only_...curve...spline.joblib` artifact is a 3D curve through
PLS1-PLS2-PLS3. The extruded surface only ever uses the curve's first two
coordinates -- PLS3 is supplied by the extrusion parameter -- so the curve can
equally well be fitted in the PLS1-PLS2 plane alone. This module fits such a
curve with the same geometric spline machinery, padding the third coordinate
with zeros so the result is an ordinary `CurveModel` that loads, saves, and
predicts like the artifact it replaces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from temporal_manifolds.viz.curve_fitting import (
    CurveModel,
    fit_geometric_spline,
    geometric_spline_endpoint_defaults,
    load_curve_model,
    serialize_curve_model,
)

DEFAULT_ENDPOINT_QUANTILE = 0.05


def _endpoints(
    xyz: np.ndarray, order_by: np.ndarray | None, quantile: float
) -> tuple[np.ndarray, np.ndarray]:
    """Pick the curve's start and end, oriented by ``order_by`` when supplied."""

    if order_by is None:
        return geometric_spline_endpoint_defaults(xyz)
    order = np.asarray(order_by, dtype=np.float64).reshape(-1)
    if len(order) != len(xyz):
        raise ValueError("The ordering values must match the number of points.")
    if not 0.0 < quantile < 0.5:
        raise ValueError("The endpoint quantile must lie strictly between 0 and 0.5.")
    low = order <= np.quantile(order, quantile)
    high = order >= np.quantile(order, 1.0 - quantile)
    if not low.any() or not high.any():
        raise ValueError("The ordering values do not separate a low and a high end.")
    # Centroids rather than single extreme points, which are noisy.
    return xyz[low].mean(axis=0), xyz[high].mean(axis=0)


def fit_plane_spline(
    points_xy: np.ndarray,
    *,
    order_by: np.ndarray | None = None,
    endpoint_quantile: float = DEFAULT_ENDPOINT_QUANTILE,
    coordinate_features: tuple[str, str, str] = ("PLS1", "PLS2", "PLS3"),
    **fit_options: Any,
) -> tuple[CurveModel, dict[str, Any]]:
    """Fit a geometric spline through 2D points, held flat in the third coordinate.

    ``order_by`` (e.g. ``log10_time_horizon_months``) orients the parameter so
    that ``t`` increases with it; without it the endpoints come from the point
    cloud's MST diameter and the direction is arbitrary.
    """

    xy = np.asarray(points_xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("Points must have shape (n, 2).")
    finite = np.isfinite(xy).all(axis=1)
    if order_by is not None:
        order = np.asarray(order_by, dtype=np.float64).reshape(-1)
        finite &= np.isfinite(order)
        order = order[finite]
    else:
        order = None
    xyz = np.column_stack([xy[finite], np.zeros(int(finite.sum()))])

    start, end = _endpoints(xyz, order, endpoint_quantile)
    # The third coordinate carries no information here; keep it exactly flat.
    start[2] = 0.0
    end[2] = 0.0

    result = fit_geometric_spline(
        xyz,
        start_point=start,
        end_point=end,
        coordinate_features=coordinate_features,
        **fit_options,
    )
    metrics = dict(result.metrics)
    metrics["plane_fit"] = True
    metrics["oriented_by_order"] = order is not None
    return result.model, metrics


def load_or_fit_plane_spline(
    path: str | Path,
    points_xy: np.ndarray,
    *,
    fit: bool = False,
    order_by: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
    **fit_options: Any,
) -> tuple[CurveModel, dict[str, Any]]:
    """Load the saved planar spline, or refit and save it when ``fit`` is true.

    Also refits when no artifact exists yet at ``path``.
    """

    path = Path(path)
    if not fit and path.is_file():
        return load_curve_model(path)

    model, metrics = fit_plane_spline(points_xy, order_by=order_by, **fit_options)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        serialize_curve_model(model, {**(metadata or {}), "fit_metrics": metrics})
    )
    return load_curve_model(path)
