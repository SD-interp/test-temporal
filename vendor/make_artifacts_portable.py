"""Rewrite curve/surface artifacts so they load under any SciPy version.

``scipy.interpolate.BSpline`` pickles a reference to a SciPy-internal module
that moves between releases, so an artifact written by SciPy 1.17 cannot be
opened by 1.18 (``No module named 'scipy._lib.array_api_compat'``). This script
replaces every pickled spline with a `PortableBSpline`, which stores only knots,
coefficients and degree.

Run it in an environment whose SciPy can still open the artifacts (the one that
wrote them), from the repo root:

    python vendor/make_artifacts_portable.py

Each rewritten file is backed up next to itself as ``<name>.scipy-pickle.bak``
unless one is already there.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "vendor") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "vendor"))

from temporal_manifolds.viz.portable_spline import portable_splines  # noqa: E402

ARTIFACTS = (
    REPO_ROOT / "models" / "ctype_only_activation_curve_PLS1-PLS2-by-t_spline.joblib",
    REPO_ROOT
    / "models"
    / "ctype_only_activation_surface_PLS1-PLS2-by-t_extruded-PLS3_degree-2.joblib",
)


def make_portable(path: Path) -> bool:
    """Rewrite one artifact in place. Returns False if it held no SciPy spline."""
    payload = joblib.load(path)
    before = repr(payload)
    payload = portable_splines(payload)
    if repr(payload) == before:
        return False

    backup = path.with_suffix(path.suffix + ".scipy-pickle.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    joblib.dump(payload, path, compress=3)
    return True


def main() -> None:
    for path in ARTIFACTS:
        if not path.exists():
            print("missing, skipped:", path.name)
            continue
        changed = make_portable(path)
        print(("rewrote " if changed else "already portable, unchanged: ") + path.name)


if __name__ == "__main__":
    main()
