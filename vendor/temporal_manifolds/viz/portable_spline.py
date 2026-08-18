"""A pickled B-spline that does not depend on the SciPy version that wrote it.

``scipy.interpolate.BSpline`` pickles a reference to a SciPy-internal helper
module (``scipy._lib.array_api_compat.numpy._aliases``), so an artifact written
by one SciPy release fails to unpickle on another with

    ModuleNotFoundError: No module named 'scipy._lib.array_api_compat'

``PortableBSpline`` stores only the knots, coefficients and degree -- plain
NumPy arrays and an int -- and rebuilds the ``BSpline`` on first use. It is a
drop-in for the spline predictors in `curve_fitting`: it is callable, and
``derivative`` returns another portable spline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import BSpline


class PortableBSpline:
    """A B-spline stored as (knots, coefficients, degree) instead of as a SciPy object."""

    def __init__(
        self,
        knots: np.ndarray,
        coefficients: np.ndarray,
        degree: int,
        *,
        extrapolate: bool = True,
    ) -> None:
        self.knots = np.asarray(knots, dtype=np.float64)
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.degree = int(degree)
        self.extrapolate = bool(extrapolate)
        self._spline: BSpline | None = None

    @classmethod
    def from_spline(cls, spline: BSpline) -> "PortableBSpline":
        """Capture a fitted SciPy spline. Accepts either attribute layout."""
        knots = getattr(spline, "t", None)
        if knots is None:
            knots = spline._t  # SciPy >= 1.17 stores the knots privately
        coefficients = getattr(spline, "c", None)
        if coefficients is None:
            coefficients = spline._c
        degree = getattr(spline, "k", None)
        if degree is None:
            degree = spline._k
        return cls(knots, coefficients, degree, extrapolate=bool(spline.extrapolate))

    @property
    def spline(self) -> BSpline:
        """The rebuilt SciPy spline, constructed once per process."""
        if self._spline is None:
            self._spline = BSpline(
                self.knots, self.coefficients, self.degree, extrapolate=self.extrapolate
            )
        return self._spline

    def __call__(self, values: Any) -> np.ndarray:
        return np.asarray(self.spline(values), dtype=np.float64)

    def derivative(self, nu: int = 1) -> "PortableBSpline":
        return PortableBSpline.from_spline(self.spline.derivative(nu))

    def antiderivative(self, nu: int = 1) -> "PortableBSpline":
        return PortableBSpline.from_spline(self.spline.antiderivative(nu))

    # The rebuilt spline is a cache, never part of the pickle.
    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_spline"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._spline = None

    def __repr__(self) -> str:
        return (
            f"PortableBSpline(degree={self.degree}, knots={self.knots.size}, "
            f"coefficients={self.coefficients.shape})"
        )


def portable_splines(value: Any) -> Any:
    """Return `value` with every SciPy ``BSpline`` inside it made portable.

    Walks dicts, sequences and object attributes, including frozen dataclasses,
    which have to be rebuilt through ``object.__setattr__``.
    """
    if isinstance(value, PortableBSpline):
        return value
    if isinstance(value, BSpline):
        return PortableBSpline.from_spline(value)
    if isinstance(value, dict):
        return {key: portable_splines(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [portable_splines(item) for item in value]
        return type(value)(converted) if not isinstance(value, tuple) else tuple(converted)
    if isinstance(value, (str, bytes, np.ndarray)) or not hasattr(value, "__dict__"):
        return value
    for name, attribute in vars(value).items():
        converted = portable_splines(attribute)
        if converted is not attribute:
            object.__setattr__(value, name, converted)
    return value
