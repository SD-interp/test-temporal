"""Classify surface coordinates and related features into temporal-scale classes."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

DEFAULT_DEGREE = 3


def build_horizon_classifier(
    *,
    degree: int = DEFAULT_DEGREE,
    interaction_only: bool = False,
    include_bias: bool = False,
    standardize: bool = True,
    C: float = 1.0,
    penalty: str | None = None,
    solver: str = "lbfgs",
    class_weight: Any = None,
    max_iter: int = 5000,
    random_state: int = 0,
) -> Pipeline:
    """Polynomial feature expansion feeding a multinomial logistic regression.

    ``degree`` sets the order of the polynomial decision surface in the input
    features; the remaining arguments expose the usual expansion and
    regularisation knobs.
    """

    if degree < 1:
        raise ValueError("The polynomial degree must be at least 1.")
    steps: list[tuple[str, Any]] = [
        (
            "poly",
            PolynomialFeatures(
                degree=degree,
                interaction_only=interaction_only,
                include_bias=include_bias,
            ),
        )
    ]
    if standardize:
        steps.append(("scale", StandardScaler()))
    # `penalty` is deprecated from scikit-learn 1.8, so only forward it when the
    # caller explicitly asks for a non-default penalty.
    logit_options: dict[str, Any] = {
        "C": C,
        "solver": solver,
        "class_weight": class_weight,
        "max_iter": max_iter,
        "random_state": random_state,
    }
    if penalty is not None:
        logit_options["penalty"] = penalty
    steps.append(("logit", LogisticRegression(**logit_options)))
    return Pipeline(steps)


def fit_horizon_classifier(
    features: np.ndarray, classes: np.ndarray, **options: Any
) -> tuple[Pipeline, dict[str, Any]]:
    """Fit the classifier on all supplied points and report training metrics.

    ``features`` is an ``(n, k)`` array of input columns, e.g. the surface
    coordinates ``t`` and ``u`` plus ``reconstruction_residual_rms``.
    """

    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("Features must have shape (n, k).")
    if not np.isfinite(features).all():
        raise ValueError("Features must be finite.")
    labels = np.asarray(classes)
    if len(features) != len(labels):
        raise ValueError("Features and labels must have equal length.")

    model = build_horizon_classifier(**options)
    model.fit(features, labels)
    predicted = model.predict(features)
    metrics = {
        "point_count": int(len(labels)),
        "input_feature_count": int(features.shape[1]),
        "class_count": int(len(np.unique(labels))),
        "feature_count": int(model.named_steps["poly"].n_output_features_),
        "accuracy": float((predicted == labels).mean()),
        "mean_absolute_class_error": float(
            np.abs(predicted.astype(float) - labels.astype(float)).mean()
        ),
    }
    return model, metrics
