"""Smoke test: load both saved models using the vendored temporal_manifolds code."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "vendor"))

from temporal_manifolds.viz.activation_pls import load_pls_model
from temporal_manifolds.viz.curve_fitting import load_curve_model

pls, pls_meta = load_pls_model(REPO_ROOT / "models/ctype_only_activation_pls_layer_out-21.joblib")
print("PLS:", pls)
print("PLS provenance:", pls_meta)

curve, curve_meta = load_curve_model(
    REPO_ROOT / "models/ctype_only_activation_curve_PLS1-PLS2-PLS3-by-t_spline.joblib"
)
print("Curve:", curve.algorithm, curve.parameter_feature, curve.coordinate_features)
print("Curve provenance:", curve_meta)
print("Curve sample:", curve.predict([0.0, 0.5, 1.0]))
