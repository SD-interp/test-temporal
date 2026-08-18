"""Load versioned Activation Atlas PLS artifacts.

Extracted from ``temporal_manifolds.viz.activation_explorer`` so that reading a
saved PLS model does not pull in torch or the activation-extraction stack.
"""

from __future__ import annotations

import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO

import joblib
import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.cross_decomposition import PLSRegression

PLS_MODEL_ARTIFACT_KIND = "temporal-manifolds.activation-pls"
PLS_MODEL_ARTIFACT_VERSION = 1


def validate_pls_model(pls: Any) -> tuple[int, int]:
    """Validate a fitted Activation Atlas PLS estimator."""
    if not isinstance(pls, PLSRegression):
        raise ValueError("The selected file does not contain a PLSRegression model.")

    components = getattr(pls, "components_", None)
    explained_variance = getattr(pls, "explained_variance_ratio_", None)
    rotations = getattr(pls, "x_rotations_", None)
    mean = getattr(pls, "mean_", None)
    x_scale = getattr(pls, "_x_std", None)
    if any(value is None for value in (components, explained_variance, rotations, mean, x_scale)):
        raise ValueError(
            "The selected PLS model is not a fitted Activation Atlas PLS artifact."
        )

    components = np.asarray(components)
    explained_variance = np.asarray(explained_variance)
    rotations = np.asarray(rotations)
    mean = np.asarray(mean)
    x_scale = np.asarray(x_scale)
    if components.ndim != 2 or not all(components.shape):
        raise ValueError("The selected PLS model has invalid fitted components.")
    component_count, feature_count = map(int, components.shape)
    if rotations.shape != (feature_count, component_count):
        raise ValueError("The selected PLS model has inconsistent rotations.")
    if explained_variance.shape != (component_count,):
        raise ValueError("The selected PLS model has invalid represented-variance metadata.")
    if mean.shape != (feature_count,) or x_scale.shape != (feature_count,):
        raise ValueError("The selected PLS model has inconsistent centering metadata.")
    if not np.isfinite(components).all() or not np.isfinite(mean).all():
        raise ValueError("The selected PLS model contains non-finite projection parameters.")
    if not np.isfinite(x_scale).all() or np.any(x_scale <= 0):
        raise ValueError("The selected PLS model has invalid activation scales.")
    fitted_feature_count = int(getattr(pls, "n_features_in_", feature_count))
    if fitted_feature_count != feature_count:
        raise ValueError("The selected PLS model has inconsistent feature metadata.")
    return component_count, feature_count


def serialize_pls_model(
    pls: PLSRegression,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> bytes:
    """Serialize a fitted PLS estimator and provenance as a versioned joblib artifact."""
    component_count, feature_count = validate_pls_model(pls)
    artifact_metadata = dict(metadata or {})
    artifact_metadata.update(
        {
            "pls_scale": bool(pls.scale),
            "pls_max_iter": int(pls.max_iter),
            "pls_tolerance": float(pls.tol),
        }
    )
    artifact = {
        "kind": PLS_MODEL_ARTIFACT_KIND,
        "version": PLS_MODEL_ARTIFACT_VERSION,
        "model": pls,
        "sklearn_version": sklearn_version,
        "component_count": component_count,
        "feature_count": feature_count,
        "metadata": artifact_metadata,
    }
    buffer = io.BytesIO()
    joblib.dump(artifact, buffer, compress=3)
    return buffer.getvalue()


def load_pls_model(
    source: str | Path | bytes | BinaryIO,
) -> tuple[PLSRegression, dict[str, Any]]:
    """Load a trusted Activation Atlas PLS artifact and return its provenance.

    Joblib and pickle files can execute arbitrary code while loading. Callers must
    only pass files from trusted sources.
    """
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    elif hasattr(source, "seek"):
        source.seek(0)
    try:
        payload = joblib.load(source)
    except Exception as exc:  # noqa: BLE001 - normalize artifact errors for UI callers
        raise ValueError(f"The PLS model file could not be loaded: {exc}") from exc

    if not isinstance(payload, Mapping) or payload.get("kind") != PLS_MODEL_ARTIFACT_KIND:
        raise ValueError("The selected file is not a supported PLS model artifact.")
    if payload.get("version") != PLS_MODEL_ARTIFACT_VERSION:
        raise ValueError(f"Unsupported PLS artifact version: {payload.get('version')!r}.")
    pls = payload.get("model")
    component_count, feature_count = validate_pls_model(pls)
    if payload.get("component_count") != component_count:
        raise ValueError("The PLS artifact's component count does not match its model.")
    if payload.get("feature_count") != feature_count:
        raise ValueError("The PLS artifact's feature count does not match its model.")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("The PLS artifact contains invalid provenance metadata.")
    provenance = {
        **dict(metadata),
        "pls_scale": bool(pls.scale),
        "pls_max_iter": int(pls.max_iter),
        "pls_tolerance": float(pls.tol),
        "artifact_kind": PLS_MODEL_ARTIFACT_KIND,
        "artifact_version": PLS_MODEL_ARTIFACT_VERSION,
        "sklearn_version": payload.get("sklearn_version"),
        "component_count": component_count,
        "feature_count": feature_count,
    }
    return pls, provenance
