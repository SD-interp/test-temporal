"""Fit, apply, and serialize curves extruded along a third chart coordinate."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import joblib
import numpy as np
from scipy import __version__ as scipy_version

from .curve_fitting import CurveModel

SURFACE_MODEL_ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class ExtrudedSurfaceModel:
    """A curve swept along one coordinate with a polynomial in-plane offset.

    The surface is ``S(t, u) = (x(t) + p0(u), y(t) + p1(u), u)`` where ``x`` and
    ``y`` are the first two coordinates of ``curve`` and ``u`` is exactly the
    third (extrusion) coordinate. ``offset_coefficients`` has shape
    ``(degree + 1, 2)`` in ascending power order, so ``u**0`` is the first row.
    """

    curve: CurveModel
    offset_coefficients: np.ndarray
    parameter_feature: str
    extrusion_feature: str
    coordinate_features: tuple[str, str, str]
    training_parameter_bounds: np.ndarray
    training_extrusion_bounds: np.ndarray
    metrics: dict[str, Any]

    @property
    def offset_degree(self) -> int:
        return int(np.asarray(self.offset_coefficients).shape[0] - 1)

    def offset(self, u: np.ndarray | list[float]) -> np.ndarray:
        """In-plane (first two coordinate) offset applied at extrusion value u."""

        values = np.asarray(u, dtype=np.float64).reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError("Extrusion values must be finite.")
        design = np.vander(values, self.offset_degree + 1, increasing=True)
        return design @ np.asarray(self.offset_coefficients, dtype=np.float64)

    def predict(
        self, parameter_values: np.ndarray | list[float], u: np.ndarray | list[float]
    ) -> np.ndarray:
        """Predict surface points for matching (t, u) pairs."""

        t = np.asarray(parameter_values, dtype=np.float64).reshape(-1)
        values = np.asarray(u, dtype=np.float64).reshape(-1)
        if t.shape != values.shape:
            raise ValueError("Parameter and extrusion values must have equal length.")
        base = self.curve.predict(t)
        points = np.empty((t.size, 3), dtype=np.float64)
        points[:, :2] = base[:, :2] + self.offset(values)
        points[:, 2] = values
        return points

    def grid(
        self, parameter_values: np.ndarray | list[float], u: np.ndarray | list[float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the surface on the outer product of t and u, for plotting."""

        t = np.asarray(parameter_values, dtype=np.float64).reshape(-1)
        values = np.asarray(u, dtype=np.float64).reshape(-1)
        base = self.curve.predict(t)
        off = self.offset(values)
        x = base[:, 0:1] + off[None, :, 0]
        y = base[:, 1:2] + off[None, :, 1]
        z = np.broadcast_to(values[None, :], x.shape).copy()
        return x, y, z

    def project(
        self, points: np.ndarray, parameter_samples: int = 4000
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (t, u, distance) of the closest surface point for each 3D point.

        Because the surface's third coordinate is the extrusion parameter, the
        closest surface point always has ``u`` equal to the point's own third
        coordinate; only ``t`` needs a search, done on a dense curve sampling.
        """

        xyz = np.asarray(points, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("Points must have shape (n, 3).")
        u = xyz[:, 2]
        t_lo, t_hi = np.asarray(self.training_parameter_bounds, dtype=np.float64)
        t_dense = np.linspace(t_lo, t_hi, int(parameter_samples))
        xy_dense = self.curve.predict(t_dense)[:, :2]
        target = xyz[:, :2] - self.offset(u)
        squared = ((target[:, None, :] - xy_dense[None, :, :]) ** 2).sum(-1)
        index = squared.argmin(1)
        distance = np.sqrt(squared[np.arange(len(index)), index])
        return t_dense[index], u, distance


def _validate_surface_model(model: Any) -> None:
    if not isinstance(model, ExtrudedSurfaceModel):
        raise ValueError("This artifact does not contain an extruded surface model.")
    coefficients = np.asarray(model.offset_coefficients, dtype=np.float64)
    if coefficients.ndim != 2 or coefficients.shape[1] != 2 or coefficients.shape[0] < 1:
        raise ValueError("Offset coefficients must have shape (degree + 1, 2).")
    if not np.isfinite(coefficients).all():
        raise ValueError("Offset coefficients must be finite.")
    if len(model.coordinate_features) != 3:
        raise ValueError("A surface needs exactly three coordinate features.")


def serialize_surface_model(
    model: ExtrudedSurfaceModel, metadata: dict[str, Any] | None = None
) -> bytes:
    """Serialize a surface and provenance in a versioned joblib artifact."""

    _validate_surface_model(model)
    payload = {
        "artifact_kind": "temporal_manifolds_extruded_surface_model",
        "artifact_version": SURFACE_MODEL_ARTIFACT_VERSION,
        "model": model,
        "metadata": dict(metadata or {}),
        "scipy_version": scipy_version,
    }
    buffer = io.BytesIO()
    joblib.dump(payload, buffer, compress=3)
    return buffer.getvalue()


def save_surface_model(
    model: ExtrudedSurfaceModel,
    destination: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a versioned surface artifact to disk and return its path."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialize_surface_model(model, metadata))
    return path


def load_surface_model(
    source: bytes | bytearray | str | Path | BinaryIO,
) -> tuple[ExtrudedSurfaceModel, dict[str, Any]]:
    """Load and validate a versioned surface artifact from bytes, a path, or a file."""

    try:
        payload = joblib.load(
            io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
        )
    except Exception as exc:  # noqa: BLE001 - normalize untrusted artifact errors
        raise ValueError(f"Surface artifact could not be loaded: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("artifact_kind") != "temporal_manifolds_extruded_surface_model"
    ):
        raise ValueError("This is not a supported extruded surface model artifact.")
    version = payload.get("artifact_version")
    if version != SURFACE_MODEL_ARTIFACT_VERSION:
        raise ValueError(f"Unsupported surface artifact version: {version!r}.")
    model = payload.get("model")
    _validate_surface_model(model)
    metadata = dict(payload.get("metadata") or {})
    metadata.update(
        {
            "artifact_kind": payload["artifact_kind"],
            "artifact_version": version,
            "scipy_version": payload.get("scipy_version"),
        }
    )
    return model, metadata
