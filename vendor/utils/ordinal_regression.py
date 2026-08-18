"""Ordinal regression models for the ordered temporal-scale classes.

The horizon classes are ordered (second-minute < minute-hour < ... < century-inf),
so predicting them as unordered labels throws away structure and treats a
one-class miss as badly as an eight-class miss. Three standard ordinal
approaches are implemented here, all sharing the same polynomial feature
expansion so they take the same tuning knobs:

``proportional_odds``
    A single latent score ``x . w`` compared against ``K - 1`` shared cutpoints
    (McCullagh's ordered logit), fitted by penalised maximum likelihood.
``binary_decomposition``
    Frank and Hall's reduction: ``K - 1`` independent binary classifiers each
    answering "is the class greater than k?", recombined into class
    probabilities.
``thresholded_regression``
    Ridge regression onto the class index, with the continuous prediction cut at
    thresholds -- either the naive half-integer grid or cutpoints tuned on the
    training labels.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, BinaryIO

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn import __version__ as sklearn_version
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ORDINAL_MODEL_ARTIFACT_VERSION = 1

ORDINAL_ALGORITHMS = (
    "proportional_odds",
    "binary_decomposition",
    "thresholded_regression",
)
DEFAULT_DEGREE = 3


class ProportionalOddsRegressor(BaseEstimator, ClassifierMixin):
    """Ordered logit: one latent score, ``K - 1`` shared cutpoints.

    ``P(y <= k) = sigmoid(theta_k - x . w)``. The cutpoints are kept increasing
    by optimising the first one directly and the gaps in log space.
    """

    def __init__(self, alpha: float = 1.0, max_iter: int = 1000, tol: float = 1e-6):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def _unpack(self, packed: np.ndarray, n_features: int) -> tuple[np.ndarray, np.ndarray]:
        weights = packed[:n_features]
        first = packed[n_features]
        gaps = np.exp(packed[n_features + 1 :])
        thresholds = np.concatenate([[first], first + np.cumsum(gaps)])
        return weights, thresholds

    def _cumulative(self, scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        """``P(y <= k)`` for every k, with implicit 0 and 1 at the ends."""

        inner = expit(thresholds[None, :] - scores[:, None])
        return np.column_stack([np.zeros(len(scores)), inner, np.ones(len(scores))])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ProportionalOddsRegressor":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) < 3:
            raise ValueError("Ordinal regression needs at least three classes.")
        index = np.searchsorted(self.classes_, y)
        n_features = X.shape[1]
        n_thresholds = len(self.classes_) - 1
        rows = np.arange(len(y))

        def objective(packed: np.ndarray) -> float:
            weights, thresholds = self._unpack(packed, n_features)
            cumulative = self._cumulative(X @ weights, thresholds)
            probability = cumulative[rows, index + 1] - cumulative[rows, index]
            # The penalty leaves the cutpoints unregularised, as usual.
            return float(
                -np.log(np.clip(probability, 1e-12, None)).sum()
                + self.alpha * (weights**2).sum()
            )

        start = np.zeros(n_features + n_thresholds)
        start[n_features] = -1.0
        start[n_features + 1 :] = np.log(2.0 / max(n_thresholds - 1, 1))
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )
        self.optimizer_result_ = result
        self.coef_, self.thresholds_ = self._unpack(result.x, n_features)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """The latent score; monotonically increasing in the class index."""

        return np.asarray(X, dtype=np.float64) @ self.coef_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        cumulative = self._cumulative(self.decision_function(X), self.thresholds_)
        return np.diff(cumulative, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(1)]


class BinaryDecompositionOrdinalRegressor(BaseEstimator, ClassifierMixin):
    """Frank and Hall: ``K - 1`` binary "is the class greater than k?" models."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 0):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BinaryDecompositionOrdinalRegressor":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        index = np.searchsorted(self.classes_, y)
        self.estimators_ = []
        for threshold in range(len(self.classes_) - 1):
            target = (index > threshold).astype(int)
            if len(np.unique(target)) < 2:
                # Degenerate split: fall back to a constant probability.
                self.estimators_.append(float(target[0]))
                continue
            model = LogisticRegression(
                C=self.C, max_iter=self.max_iter, random_state=self.random_state
            )
            model.fit(X, target)
            self.estimators_.append(model)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        greater = np.column_stack(
            [
                np.full(len(X), estimator)
                if isinstance(estimator, float)
                else estimator.predict_proba(X)[:, 1]
                for estimator in self.estimators_
            ]
        )
        # Enforce monotonicity: P(y > k) cannot rise with k.
        greater = np.minimum.accumulate(greater, axis=1)
        cumulative = np.column_stack([np.ones(len(X)), greater, np.zeros(len(X))])
        return np.clip(-np.diff(cumulative, axis=1), 0.0, None)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probability = self.predict_proba(X)
        total = probability.sum(1, keepdims=True)
        normalized = np.divide(probability, np.where(total > 0, total, 1))
        return self.classes_[normalized.argmax(1)]


class ThresholdedRegressor(BaseEstimator, ClassifierMixin):
    """Ridge regression onto the class index, cut at ordered thresholds.

    With ``tune_thresholds`` the cutpoints are chosen by a coordinate sweep over
    the range of the training predictions to maximise training accuracy;
    otherwise the naive half-integer grid is used.
    """

    def __init__(
        self, alpha: float = 1.0, tune_thresholds: bool = True, threshold_grid: int = 200
    ):
        self.alpha = alpha
        self.tune_thresholds = tune_thresholds
        self.threshold_grid = threshold_grid

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ThresholdedRegressor":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        index = np.searchsorted(self.classes_, y).astype(np.float64)
        self.regressor_ = Ridge(alpha=self.alpha).fit(X, index)
        predicted = self.regressor_.predict(X)
        self.thresholds_ = (np.arange(1, len(self.classes_)) - 0.5).astype(np.float64)
        if self.tune_thresholds:
            candidates = np.linspace(
                predicted.min(), predicted.max(), int(self.threshold_grid)
            )
            thresholds = self.thresholds_.copy()
            for _ in range(3):  # a few sweeps are enough to settle
                for position in range(len(thresholds)):
                    lower = thresholds[position - 1] if position else -np.inf
                    upper = (
                        thresholds[position + 1]
                        if position + 1 < len(thresholds)
                        else np.inf
                    )
                    allowed = candidates[(candidates > lower) & (candidates < upper)]
                    if not len(allowed):
                        continue
                    best, best_score = thresholds[position], -np.inf
                    for candidate in allowed:
                        thresholds[position] = candidate
                        score = (self._cut(predicted, thresholds) == index).mean()
                        if score > best_score:
                            best, best_score = candidate, score
                    thresholds[position] = best
            self.thresholds_ = thresholds
        return self

    @staticmethod
    def _cut(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        return np.searchsorted(thresholds, values, side="right").astype(np.float64)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.regressor_.predict(np.asarray(X, dtype=np.float64))

    def predict(self, X: np.ndarray) -> np.ndarray:
        cut = self._cut(self.decision_function(X), self.thresholds_).astype(int)
        return self.classes_[np.clip(cut, 0, len(self.classes_) - 1)]


def build_ordinal_model(
    *,
    algorithm: str = "proportional_odds",
    degree: int = DEFAULT_DEGREE,
    interaction_only: bool = False,
    include_bias: bool = False,
    standardize: bool = True,
    **estimator_options: Any,
) -> Pipeline:
    """Polynomial expansion feeding one of the ordinal estimators."""

    if algorithm not in ORDINAL_ALGORITHMS:
        raise ValueError(
            f"Unknown ordinal algorithm {algorithm!r}; expected one of {ORDINAL_ALGORITHMS}."
        )
    if degree < 1:
        raise ValueError("The polynomial degree must be at least 1.")

    estimators = {
        "proportional_odds": ProportionalOddsRegressor,
        "binary_decomposition": BinaryDecompositionOrdinalRegressor,
        "thresholded_regression": ThresholdedRegressor,
    }
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
    steps.append(("ordinal", estimators[algorithm](**estimator_options)))
    return Pipeline(steps)


def ordinal_metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Accuracy plus the ordinal-aware error measures."""

    true = np.asarray(true, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    error = predicted - true
    return {
        "accuracy": float((error == 0).mean()),
        "within_one_accuracy": float((np.abs(error) <= 1).mean()),
        "mean_absolute_class_error": float(np.abs(error).mean()),
        "rmse_class_error": float(np.sqrt((error**2).mean())),
    }


def fit_ordinal_model(
    features: np.ndarray, classes: np.ndarray, **options: Any
) -> tuple[Pipeline, dict[str, Any]]:
    """Fit an ordinal model on all supplied points and report training metrics."""

    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("Features must have shape (n, k).")
    if not np.isfinite(features).all():
        raise ValueError("Features must be finite.")
    labels = np.asarray(classes)
    if len(features) != len(labels):
        raise ValueError("Features and labels must have equal length.")

    model = build_ordinal_model(**options)
    model.fit(features, labels)
    predicted = model.predict(features)
    metrics: dict[str, Any] = {
        "algorithm": options.get("algorithm", "proportional_odds"),
        "point_count": int(len(labels)),
        "input_feature_count": int(features.shape[1]),
        "expanded_feature_count": int(model.named_steps["poly"].n_output_features_),
        "class_count": int(len(np.unique(labels))),
    }
    metrics.update(ordinal_metrics(labels, predicted))
    return model, metrics


def save_ordinal_model(
    model: Pipeline,
    destination: str | Path,
    feature_names: list[str],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a versioned ordinal-model artifact, with the feature order it expects.

    The feature names travel with the model so a prediction-only notebook can
    build its design matrix in the right order instead of hard-coding it.
    """

    if not isinstance(model, Pipeline) or "ordinal" not in model.named_steps:
        raise ValueError("Expected a fitted ordinal pipeline.")
    names = [str(name) for name in feature_names]
    if not names:
        raise ValueError("At least one feature name is required.")
    payload = {
        "artifact_kind": "temporal_manifolds_ordinal_model",
        "artifact_version": ORDINAL_MODEL_ARTIFACT_VERSION,
        "model": model,
        "feature_names": names,
        "classes": np.asarray(model.named_steps["ordinal"].classes_),
        "metadata": dict(metadata or {}),
        "sklearn_version": sklearn_version,
    }
    buffer = io.BytesIO()
    joblib.dump(payload, buffer, compress=3)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.getvalue())
    return path


def load_ordinal_model(
    source: bytes | bytearray | str | Path | BinaryIO,
) -> tuple[Pipeline, list[str], dict[str, Any]]:
    """Load a versioned ordinal artifact as (model, feature names, metadata)."""

    try:
        payload = joblib.load(
            io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
        )
    except Exception as exc:  # noqa: BLE001 - normalize untrusted artifact errors
        raise ValueError(f"Ordinal artifact could not be loaded: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("artifact_kind") != "temporal_manifolds_ordinal_model"
    ):
        raise ValueError("This is not a supported ordinal model artifact.")
    version = payload.get("artifact_version")
    if version != ORDINAL_MODEL_ARTIFACT_VERSION:
        raise ValueError(f"Unsupported ordinal artifact version: {version!r}.")
    model = payload.get("model")
    if not isinstance(model, Pipeline) or "ordinal" not in model.named_steps:
        raise ValueError("The artifact does not contain a fitted ordinal pipeline.")
    feature_names = [str(name) for name in payload.get("feature_names") or []]
    if not feature_names:
        raise ValueError("The artifact does not record its feature order.")
    metadata = dict(payload.get("metadata") or {})
    metadata.update(
        {
            "artifact_kind": payload["artifact_kind"],
            "artifact_version": version,
            "sklearn_version": payload.get("sklearn_version"),
            "classes": np.asarray(payload.get("classes")).tolist(),
        }
    )
    return model, feature_names, metadata
