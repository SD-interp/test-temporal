"""Residual-PCA artifact: the basis the horizon classifier's last three features live in.

The Activation Atlas app fits this PCA on the fly (see
``reconstruction_residual_statistics`` in the source project) and never saved it,
so ``fit_residual_pca.py`` recovers it from the cached activations. This module
only stores, loads and applies it.

A point's residual is what the retained PLS components failed to reconstruct:
``residual = activation - pls.inverse_transform(pls.transform(activation))``.
Its first three principal components are ``reconstruction_residual_PC1..3``.
"""

from __future__ import annotations

import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO

import joblib
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import IncrementalPCA

RESIDUAL_PCA_ARTIFACT_KIND = "temporal-manifolds.reconstruction-residual-pca"
RESIDUAL_PCA_ARTIFACT_VERSION = 1

RESIDUAL_FEATURE_NAMES = (
    "reconstruction_residual_PC1",
    "reconstruction_residual_PC2",
    "reconstruction_residual_PC3",
)


def save_residual_pca(
    residual_pca: IncrementalPCA,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write a fitted residual PCA as a versioned joblib artifact."""
    path = Path(path)
    artifact = {
        "kind": RESIDUAL_PCA_ARTIFACT_KIND,
        "version": RESIDUAL_PCA_ARTIFACT_VERSION,
        "model": residual_pca,
        "component_count": int(residual_pca.components_.shape[0]),
        "feature_count": int(residual_pca.components_.shape[1]),
        "metadata": dict(metadata or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path, compress=3)
    return path


def load_residual_pca(
    source: str | Path | bytes | BinaryIO,
) -> tuple[IncrementalPCA, dict[str, Any]]:
    """Load a trusted residual-PCA artifact and return it with its provenance.

    Joblib files can execute arbitrary code while loading; only pass trusted files.
    """
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    try:
        payload = joblib.load(source)
    except Exception as exc:  # noqa: BLE001 - normalize artifact errors
        raise ValueError(f"The residual PCA file could not be loaded: {exc}") from exc

    if not isinstance(payload, Mapping) or payload.get("kind") != RESIDUAL_PCA_ARTIFACT_KIND:
        raise ValueError("The selected file is not a residual PCA artifact.")
    if payload.get("version") != RESIDUAL_PCA_ARTIFACT_VERSION:
        raise ValueError(f"Unsupported residual PCA version: {payload.get('version')!r}.")

    residual_pca = payload.get("model")
    if not isinstance(residual_pca, IncrementalPCA):
        raise ValueError("The residual PCA artifact does not contain a fitted IncrementalPCA.")
    return residual_pca, dict(payload.get("metadata", {}))


def reconstruction_residual(pls: PLSRegression, activations: np.ndarray) -> np.ndarray:
    """Return the part of each activation the PLS components cannot reconstruct."""
    activations = np.atleast_2d(np.asarray(activations, dtype=np.float64))
    scores = pls.transform(activations)
    return activations - np.asarray(pls.inverse_transform(scores), dtype=np.float64)


def project_activations(
    pls: PLSRegression,
    residual_pca: IncrementalPCA,
    activations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return PLS scores, residual PC scores and residual RMS for each activation.

    ``activations`` is a single vector or a matrix of rows in activation space.
    """
    activations = np.atleast_2d(np.asarray(activations, dtype=np.float64))
    scores = np.asarray(pls.transform(activations), dtype=np.float64)
    residuals = activations - np.asarray(pls.inverse_transform(scores), dtype=np.float64)
    residual_scores = np.asarray(residual_pca.transform(residuals), dtype=np.float64)
    residual_rms = np.sqrt(np.mean(np.square(residuals), axis=1))
    return scores, residual_scores, residual_rms
