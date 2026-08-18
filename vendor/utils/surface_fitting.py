"""Fit a curve extruded along a third coordinate to a 3D point cloud."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from temporal_manifolds.viz.curve_fitting import CurveModel
from temporal_manifolds.viz.extruded_surface import (
    ExtrudedSurfaceModel,
    load_surface_model,
    save_surface_model,
)

DEFAULT_OFFSET_DEGREE = 2
DEFAULT_PARAMETER_SAMPLES = 4000
DEFAULT_MAX_ITERATIONS = 50
# Bound extension: each step grows one end by this fraction of the curve's own
# training span, and a final margin keeps the data off the very endpoints.
DEFAULT_EXTENSION_STEP = 0.05
DEFAULT_MAX_EXTENSIONS = 200
DEFAULT_EXTENSION_MARGIN = 0.01


def _design(u: np.ndarray, degree: int) -> np.ndarray:
    """Polynomial design matrix in ascending power order."""

    return np.vander(u, degree + 1, increasing=True)


def _nearest_parameter(
    target_xy: np.ndarray, xy_dense: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    squared = ((target_xy[:, None, :] - xy_dense[None, :, :]) ** 2).sum(-1)
    index = squared.argmin(1)
    return index, np.sqrt(squared[np.arange(len(index)), index])


def extend_parameter_bounds(
    curve: CurveModel,
    target_xy: np.ndarray,
    *,
    bounds: np.ndarray | tuple[float, float] | None = None,
    parameter_samples: int = DEFAULT_PARAMETER_SAMPLES,
    step_fraction: float = DEFAULT_EXTENSION_STEP,
    max_extensions: int = DEFAULT_MAX_EXTENSIONS,
    margin: float = DEFAULT_EXTENSION_MARGIN,
) -> tuple[float, float]:
    """Grow the parameter range until no point projects onto an endpoint.

    A point whose nearest curve sample is the first or last one is really
    projecting *past* that end: the curve stops before the data does, and the
    clamped projection understates its distance. Each such end is pushed out
    repeatedly (the curve's predictors extrapolate along the endpoint tangents)
    until every point lands strictly inside, then a small margin is added.
    """

    lo, hi = np.asarray(
        curve.training_parameter_bounds if bounds is None else bounds, dtype=np.float64
    )
    step = float(step_fraction) * (hi - lo)
    if step <= 0:
        raise ValueError("The extension step must be positive.")

    for _ in range(int(max_extensions)):
        t_dense = np.linspace(lo, hi, int(parameter_samples))
        index, _ = _nearest_parameter(target_xy, curve.predict(t_dense)[:, :2])
        below = bool((index == 0).any())
        above = bool((index == len(t_dense) - 1).any())
        if not below and not above:
            break
        if below:
            lo -= step
        if above:
            hi += step

    pad = float(margin) * (hi - lo)
    return float(lo - pad), float(hi + pad)


def fit_extruded_surface(
    points: np.ndarray,
    curve: CurveModel,
    *,
    offset_degree: int = DEFAULT_OFFSET_DEGREE,
    parameter_samples: int = DEFAULT_PARAMETER_SAMPLES,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    extrusion_feature: str = "PLS3",
    extend_bounds: bool = True,
    extension_step: float = DEFAULT_EXTENSION_STEP,
    max_extensions: int = DEFAULT_MAX_EXTENSIONS,
    extension_margin: float = DEFAULT_EXTENSION_MARGIN,
) -> ExtrudedSurfaceModel:
    """Fit ``S(t, u) = (x(t) + p0(u), y(t) + p1(u), u)`` by geometric RMSE.

    The surface's third coordinate is the extrusion parameter ``u``, so the
    closest surface point to a datum necessarily has ``u`` equal to that datum's
    third coordinate. Geometric distance therefore reduces to the in-plane
    distance minimised over ``t``, and the polynomial offsets are fitted by
    alternating least squares: assign each point its nearest ``t`` on the
    current offset curve, then refit the offsets in closed form.

    With ``extend_bounds`` the parameter range is first widened (via the curve's
    tangent extrapolation) until the curve spans the whole data range, so no
    point projects onto a clamped endpoint. The fitted range is stored on the
    returned model as ``training_parameter_bounds``.
    """

    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("Points must have shape (n, 3).")
    if not np.isfinite(xyz).all():
        raise ValueError("Points must be finite.")
    if offset_degree < 0:
        raise ValueError("The offset degree must be non-negative.")

    u = xyz[:, 2]
    design = _design(u, offset_degree)

    curve_bounds = np.asarray(curve.training_parameter_bounds, dtype=np.float64)
    t_lo, t_hi = curve_bounds

    def sampled(lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
        t_dense = np.linspace(lo, hi, int(parameter_samples))
        return t_dense, curve.predict(t_dense)[:, :2]

    def extended(target_xy: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
        if not extend_bounds:
            return lo, hi
        return extend_parameter_bounds(
            curve,
            target_xy,
            bounds=(lo, hi),
            parameter_samples=parameter_samples,
            step_fraction=extension_step,
            max_extensions=max_extensions,
            margin=extension_margin,
        )

    # Extend before fitting so the curve already spans the data; the offsets
    # shift the targets as they change, so re-check the span each iteration.
    t_lo, t_hi = extended(xyz[:, :2], t_lo, t_hi)
    t_dense, xy_dense = sampled(t_lo, t_hi)

    coefficients = np.zeros((offset_degree + 1, 2), dtype=np.float64)
    for _ in range(int(max_iterations)):
        target = xyz[:, :2] - design @ coefficients
        new_lo, new_hi = extended(target, t_lo, t_hi)
        if (new_lo, new_hi) != (t_lo, t_hi):
            t_lo, t_hi = new_lo, new_hi
            t_dense, xy_dense = sampled(t_lo, t_hi)
        index, _ = _nearest_parameter(target, xy_dense)
        residual = xyz[:, :2] - xy_dense[index]
        updated, *_ = np.linalg.lstsq(design, residual, rcond=None)
        converged = np.allclose(updated, coefficients, atol=1e-12)
        coefficients = updated
        if converged:
            break

    # Re-derive the span from the curve's own bounds rather than keeping the
    # accumulated one: intermediate offsets can push the ends out further than
    # the converged fit needs, and a tighter span samples the curve more finely.
    target = xyz[:, :2] - design @ coefficients
    t_lo, t_hi = extended(target, *curve_bounds)
    t_dense, xy_dense = sampled(t_lo, t_hi)
    _, distance = _nearest_parameter(target, xy_dense)
    _, straight_distance = _nearest_parameter(xyz[:, :2], xy_dense)

    return ExtrudedSurfaceModel(
        curve=curve,
        offset_coefficients=coefficients,
        parameter_feature=curve.parameter_feature,
        extrusion_feature=extrusion_feature,
        coordinate_features=tuple(curve.coordinate_features),
        training_parameter_bounds=np.asarray([t_lo, t_hi], dtype=np.float64),
        training_extrusion_bounds=np.asarray([u.min(), u.max()], dtype=np.float64),
        metrics={
            "point_count": int(len(xyz)),
            "offset_degree": int(offset_degree),
            "curve_training_parameter_bounds": [float(curve_bounds[0]), float(curve_bounds[1])],
            "parameter_bounds": [float(t_lo), float(t_hi)],
            "parameter_bounds_extended": bool(
                not np.allclose([t_lo, t_hi], curve_bounds)
            ),
            "geometric_rmse": float(np.sqrt((distance**2).mean())),
            "geometric_rmse_straight_extrusion": float(
                np.sqrt((straight_distance**2).mean())
            ),
            "geometric_max_distance": float(distance.max()),
        },
    )


def load_or_fit_extruded_surface(
    path: str | Path,
    points: np.ndarray,
    curve: CurveModel,
    *,
    fit: bool = False,
    metadata: dict[str, Any] | None = None,
    **fit_options: Any,
) -> tuple[ExtrudedSurfaceModel, dict[str, Any]]:
    """Load the saved surface, or refit and save it when ``fit`` is true.

    Also refits when no artifact exists yet at ``path``.
    """

    path = Path(path)
    if not fit and path.is_file():
        return load_surface_model(path)

    model = fit_extruded_surface(points, curve, **fit_options)
    save_surface_model(model, path, metadata)
    return load_surface_model(path)
