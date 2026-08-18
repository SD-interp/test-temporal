"""Recover the reconstruction-residual PCA from the cached activation batches.

The horizon classifier's last three features are principal components of the
PLS reconstruction residual. The Activation Atlas app fit that PCA on the fly
and only wrote the resulting scores into the projection CSV, so it has to be
rebuilt here before a new prompt can be scored.

The rebuild mirrors the app: activations are averaged per
``(source folder, task, time horizon)`` group -- the same aggregation that
produced each CSV row -- the PLS reconstruction is subtracted, and an
``IncrementalPCA`` is fit over the residuals in CSV row order.

Run from the repo root:

    python vendor/activations/fit_residual_pca.py
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "vendor") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "vendor"))

from activations.residual_pca import save_residual_pca  # noqa: E402
from temporal_manifolds.viz.activation_pls import load_pls_model  # noqa: E402
from utils.horizon_classes import UNIT_TO_MONTHS  # noqa: E402

ACTIVATION_ROOT = REPO_ROOT / ".acts"
PROJECTION_CSV = REPO_ROOT / "data" / "task_sf_split_activation_pls_projection.csv"
PLS_PATH = REPO_ROOT / "models" / "ctype_only_activation_pls_layer_out-21.joblib"
OUTPUT_PATH = REPO_ROOT / "models" / "ctype_only_residual_pca_layer_out-21.joblib"

LAYER_COMPONENT = "layer_out/21"
RESIDUAL_COMPONENTS = 3
BATCH_SIZE = 2048

# Reading `.acts/` needs torch; fitting needs the same scikit-learn the other
# artifacts were pickled with. When one interpreter has neither, run
# `--write-residuals` where torch is available and `--read-residuals` where
# scikit-learn matches.
RESIDUAL_CACHE = ACTIVATION_ROOT / "_reconstruction_residuals.npy"

# `base_unit` is plural in the cached metadata, and the horizon table has no
# millennium entry.
UNIT_MONTHS = dict(UNIT_TO_MONTHS) | {"millennium": 12000.0}
UNIT_ALIASES = {"millennia": "millennium", "centuries": "century"}


def _horizon_months(base_value: float, base_unit: str) -> float:
    unit = UNIT_ALIASES.get(base_unit, base_unit).rstrip("s")
    return float(base_value) * UNIT_MONTHS[unit]


def group_mean_activations(folder: Path) -> dict[tuple[str, float], np.ndarray]:
    """Average one folder's cached activations per (task, horizon) group."""
    # Imported here so the fitting half of this script runs in an environment
    # without torch, reading a cached residual matrix instead.
    import torch

    sums: dict[tuple[str, float], np.ndarray] = {}
    counts: dict[tuple[str, float], int] = {}
    for path in sorted(glob.glob(str(folder / "*.pt"))):
        batch = torch.load(path, map_location="cpu", weights_only=False)
        if batch["layer_component"] != LAYER_COMPONENT:
            raise ValueError(f"{path} caches {batch['layer_component']!r}.")
        values = batch["activations"][LAYER_COMPONENT][:, 0, :].to(torch.float64).numpy()
        for index, metadata in enumerate(batch["prompt_metadata"]):
            key = (
                metadata["task"],
                _horizon_months(metadata["base_value"], metadata["base_unit"]),
            )
            if key in sums:
                sums[key] += values[index]
                counts[key] += 1
            else:
                sums[key] = values[index].copy()
                counts[key] = 1
    return {key: total / counts[key] for key, total in sums.items()}


def aggregated_activation_matrix(projection: pd.DataFrame) -> np.ndarray:
    """Return one averaged activation per projection row, in CSV row order."""
    means = {
        folder: group_mean_activations(ACTIVATION_ROOT / folder)
        for folder in sorted(projection["source_folder"].unique())
    }
    width = len(next(iter(next(iter(means.values())).values())))
    matrix = np.empty((len(projection), width), dtype=np.float64)
    for row_index, row in enumerate(projection.itertuples()):
        folder_means = means[row.source_folder]
        keys = [key for key in folder_means if key[0] == row.task]
        if not keys:
            raise KeyError(f"No cached activations for task {row.task!r}.")
        horizons = np.array([key[1] for key in keys])
        # The CSV stores rounded horizons, so match the nearest cached level.
        key = keys[int(np.argmin(np.abs(np.log(horizons) - np.log(row.time_horizon_months))))]
        matrix[row_index] = folder_means[key]
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-residuals",
        action="store_true",
        help=f"compute the residual matrix, cache it at {RESIDUAL_CACHE.name} and stop",
    )
    parser.add_argument(
        "--read-residuals",
        action="store_true",
        help="fit and save from the cached residual matrix instead of reading .acts/",
    )
    arguments = parser.parse_args()

    projection = pd.read_csv(PROJECTION_CSV)
    pls, pls_metadata = load_pls_model(PLS_PATH)

    if arguments.read_residuals:
        residuals = np.load(RESIDUAL_CACHE)
        if len(residuals) != len(projection):
            raise SystemExit(
                f"{RESIDUAL_CACHE.name} has {len(residuals)} rows but the projection has "
                f"{len(projection)}; recompute it with --write-residuals."
            )
    else:
        activations = aggregated_activation_matrix(projection)
        scores = projection[["PLS1", "PLS2", "PLS3"]].to_numpy(float)
        residuals = activations - np.asarray(pls.inverse_transform(scores), dtype=np.float64)
        if arguments.write_residuals:
            np.save(RESIDUAL_CACHE, residuals)
            print("wrote", RESIDUAL_CACHE)
            return

    batches = [
        slice(start, min(start + BATCH_SIZE, len(residuals)))
        for start in range(0, len(residuals), BATCH_SIZE)
    ]
    residual_pca = IncrementalPCA(n_components=RESIDUAL_COMPONENTS, batch_size=BATCH_SIZE)
    for batch in batches:
        residual_pca.partial_fit(residuals[batch])
    recovered = np.vstack([residual_pca.transform(residuals[batch]) for batch in batches])

    # The CSV's own residual PCs are the reference: agreement confirms the
    # aggregation and the PCA fit order were reproduced.
    stored = projection[
        [f"reconstruction_residual_PC{index + 1}" for index in range(RESIDUAL_COMPONENTS)]
    ].to_numpy(float)
    correlations = [
        float(np.corrcoef(recovered[:, index], stored[:, index])[0, 1])
        for index in range(RESIDUAL_COMPONENTS)
    ]
    print("correlation with the stored residual PCs:", np.round(correlations, 6))
    if min(correlations) < 0.999:
        raise SystemExit("Recovered residual PCs do not match the projection CSV.")

    path = save_residual_pca(
        residual_pca,
        OUTPUT_PATH,
        metadata={
            "layer_component": LAYER_COMPONENT,
            "cached_position": pls_metadata.get("cached_position"),
            "pls_artifact": PLS_PATH.name,
            "source_data": str(PROJECTION_CSV.relative_to(REPO_ROOT)).replace("\\", "/"),
            "point_count": int(len(residuals)),
            "batch_size": BATCH_SIZE,
            "feature_names": [
                f"reconstruction_residual_PC{index + 1}" for index in range(RESIDUAL_COMPONENTS)
            ],
            "stored_pc_correlations": correlations,
        },
    )
    print("wrote", path)


if __name__ == "__main__":
    main()
